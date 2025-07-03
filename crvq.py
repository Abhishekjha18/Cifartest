import numpy as np
import torch
import torch.nn as nn
from modules import *

class CRVQ:
    """
    Channel-Relaxed Vector Quantization (CRVQ) implementation.

    This class implements the CRVQ algorithm for quantizing neural network weights,
    with special handling for important channels using extended codebooks.
    Handles both Linear and Convolutional layers.
    """

    def __init__(self, m=4, lambda_ratio=0.3, d=8, e=8, loss_threshold=1e-4):
        """
        Initialize CRVQ quantizer.

        Args:
            m (int): Number of codebooks (1 basic + m-1 extended)
            lambda_ratio (float): Ratio of important channels (0.0 to 1.0)
            d (int): Vector dimension for quantization
            e (int): Codebook bit-width (2^e entries in each codebook)
            loss_threshold (float): Loss threshold for fine-tuning convergence
        """
        self.m = m
        self.lambda_ratio = lambda_ratio
        self.d = d
        self.e = e
        self.loss_threshold = loss_threshold
        self.codebooks = {}
        self.encodings = {}
        self.channel_mappings = {} # Stores mapping from sorted back to original indices

    def compute_hessian_proxy(self, X):
        """
        Compute Hessian proxy XXT from activations.

        Args:
            X (np.array): Activation matrix (N x O) where N is input features, O is observations

        Returns:
            np.array: Hessian proxy matrix (N x N)
        """
        # Ensure X has enough observations (columns) for meaningful XXT
        if X.shape[1] < X.shape[0]:
             print(f"Warning: Activation matrix X has shape {X.shape}. Not enough observations (columns) for robust XXT calculation.")
             # Use a smaller subset of X if needed or suggest more calibration data.
             # For now, proceed but results might be less reliable.

        # Compute XXT. If X is (N x O), XXT is (N x N).
        # XXT = X @ X.T
        # If X is already N x O, X.T is O x N, (N x O) @ (O x N) -> (N x N)
        # This aligns with the second dimension of the reshaped weight matrix W_reshaped (M x N).
        return np.dot(X, X.T)


    def quantize_layer(self, W, X=None, layer_name="layer", block_idx=-1, model=None, calibration_sample=None):
        """
        Quantize a single layer using CRVQ algorithm.

        Args:
            W (np.array): Weight matrix to quantize (M x N for Linear, O x I x kH x kW for Conv)
            X (np.array): Activation input for this layer (N_flat x O), where N_flat is the
                          flattened input dimension to the matrix multiplication.
            layer_name (str): Name identifier for this layer
            block_idx (int): Index of the block/layer being quantized
            model (nn.Module): The entire model (needed for block fine-tuning)
            calibration_sample (tuple): A sample input/target pair (X, y) for block fine-tuning

        Returns:
            tuple: (quantized_weights, quantization_info)
        """
        print(f"\n=== Quantizing {layer_name} ===")

        original_shape = W.shape
        is_conv = len(original_shape) == 4

        # Convert torch tensor to numpy if needed
        if torch.is_tensor(W):
            W = W.detach().cpu().numpy()
        if X is not None and torch.is_tensor(X):
            X = X.detach().cpu().numpy()
            # If X is batch x features, transpose to features x batch (N_flat x O)
            if X.shape[0] > X.shape[1] and X.shape[0] == W.shape[1]: # Heuristic: assuming batch-first and batch > features
                 X = X.T


        # Reshape W to 2D for processing (M x N_flat)
        if is_conv:
            W_reshaped = W.reshape(original_shape[0], -1)
        else:
            W_reshaped = W.copy() # Already 2D

        # Step 1: Pre-quantize weights to estimate quantization error
        # Prequantize operates on the reshaped 2D weight matrix
        W_reshaped_prequant = Prequantize(W_reshaped, self.d, self.e)

        # Step 2: Compute Hessian proxy if activations provided
        if X is not None:
            # Ensure X's first dimension matches the second dimension of W_reshaped
            if X.shape[0] == W_reshaped.shape[1]:
                 XXT = self.compute_hessian_proxy(X)
            else:
                 print(f"Warning: Activation dimension {X.shape[0]} does not match reshaped weight dimension {W_reshaped.shape[1]}. Cannot compute accurate Hessian proxy. Using identity matrix.")
                 # Use identity matrix as fallback
                 XXT = np.eye(W_reshaped.shape[1])
        else:
            print("Warning: No activations provided. Cannot compute Hessian proxy. Using identity matrix.")
            # Use identity matrix as fallback
            XXT = np.eye(W_reshaped.shape[1])


        # Step 3: Compute channel importance
        # Importance is computed based on the reshaped 2D matrix.
        # ComputeImportance now aims to return importance per row (output channel).
        # NOTE: The original ComputeImportance logic was for columns (input features).
        # This needs careful alignment with how lambda_ratio selects channels (rows vs columns).
        # Assuming lambda_ratio applies to *output channels* (rows) for both linear and conv.
        # Therefore, ComputeImportance should produce importance per row (M values).
        # REVISING ComputeImportance to calculate importance per row.
        importance = ComputeImportance(W_reshaped, W_reshaped_prequant, XXT) # Need to modify ComputeImportance

        # Step 4: Reorder channels (rows) based on importance
        # ReorderChannels operates on the reshaped 2D matrix (M rows, N_flat columns)
        W_reshaped_sorted, channel_mapping = ReorderChannels(W_reshaped, importance)
        self.channel_mappings[layer_name] = channel_mapping # Mapping applies to original rows

        # Step 5: Determine important channels (rows)
        M, N_flat = W_reshaped.shape
        n_important = int(self.lambda_ratio * M)
        n_important = max(1, n_important) # Ensure at least one important channel
        n_important = min(n_important, M) # Don't exceed total channels

        print(f"  Identified {n_important} important channels out of {M}")

        # Step 6: Create basic codebook and perform initial quantization on the sorted reshaped matrix
        vectors_sorted, padding = partition_to_vectors(W_reshaped_sorted, self.d)
        C_base = KMeansCodebook(vectors_sorted, n_clusters=2**self.e)
        W_encoded_vectors_sorted, B_base = VectorQuant(vectors_sorted, C_base)

        # Convert back to matrix form (still sorted and padded)
        W_encoded_reshaped_padded_sorted = vectors_to_matrix(W_encoded_vectors_sorted, W_reshaped_sorted.shape, padding)

        # Remove padding from the encoded sorted reshaped matrix
        if padding > 0:
            W_encoded_reshaped_sorted = W_encoded_reshaped_padded_sorted[:, :-padding]
        else:
            W_encoded_reshaped_sorted = W_encoded_reshaped_padded_sorted


        # Store codebooks and encodings (related to sorted vectors)
        # Encodings are for the vectors (total_vectors, )
        self.codebooks[layer_name] = {'base': C_base, 'extended': []}
        self.encodings[layer_name] = {'base': B_base, 'extended': []}


        # Step 7: Extended codebook fitting for important channels (rows)
        # This additive quantization operates on the *rows* (output channels) of the reshaped sorted matrix.
        W_lambda_sorted = W_reshaped_sorted[:n_important, :]  # Important rows (channels)
        W_lambda_encoded_sorted = W_encoded_reshaped_sorted[:n_important, :] # Initial encoding of important rows

        # Keep track of original shapes for converting vectors back
        original_lambda_shape = W_lambda_sorted.shape

        extended_encodings = []

        if self.m > 1 and n_important > 0:
            for t in range(self.m - 1):
                print(f"  Fitting extended codebook {t+1} for important channels...")

                # Compute residual error for important rows
                E_t_sorted = ComputeError(W_lambda_sorted, W_lambda_encoded_sorted)

                # Partition residual error matrix (of important rows) into vectors
                E_t_vectors_sorted, et_padding = partition_to_vectors(E_t_sorted, self.d)

                # Create extended codebook for the residual vectors
                C_t_ext = KMeansCodebook(E_t_vectors_sorted, n_clusters=2**self.e)

                # Quantize residual vectors with the extended codebook
                E_t_encoded_vectors_sorted, B_t_ext = VectorQuant(E_t_vectors_sorted, C_t_ext)
                E_t_encoded_sorted = vectors_to_matrix(E_t_encoded_vectors_sorted, original_lambda_shape, et_padding)

                # Update quantized important channels additively
                W_lambda_encoded_sorted = Update(W_lambda_encoded_sorted, E_t_encoded_sorted)

                # Store extended codebook and encoding
                self.codebooks[layer_name]['extended'].append(C_t_ext)
                # B_t_ext are encodings for vectors from E_t_sorted (important rows only)
                extended_encodings.append(B_t_ext)


            # Update the full encoded matrix with the final additive result for important rows
            W_encoded_reshaped_sorted[:n_important, :] = W_lambda_encoded_sorted
            self.encodings[layer_name]['extended'] = extended_encodings


        # Step 8: Fine-tuning (optional)
        # Fine-tuning should operate on the sorted reshaped weights and corresponding activations
        if X is not None:
            print("  Fine-tuning codebooks...")
            # Need to generate W_encoded_reshaped from codebooks and encodings for the current state
            # This is complex as base and extended encodings combine additively.
            # For simplicity, we'll fine-tune based on the current state of W_encoded_reshaped_sorted
            # and assume the codebooks are updated. A proper implementation needs to reconstruct
            # W_encoded_reshaped_sorted from the updated codebooks and current encodings.

            # This calls a FineTuneCodebook placeholder.
            FineTuneCodebook(
                self.codebooks[layer_name]['base'],
                self.codebooks[layer_name]['extended'],
                W_reshaped_sorted, # Use sorted original for loss calc
                W_encoded_reshaped_sorted, # Use current encoded state for loss calc
                X # Activations
            )

            # Beam search optimization (placeholder)
            # BeamSearchOptimize should update the *encodings* (B_base, B_ext_list)
            # based on the updated codebooks.
            # After BeamSearchOptimize, W_encoded_reshaped_sorted needs to be reconstructed
            # from the updated encodings and codebooks.
            # Skipping actual beam search for now.

            # B_base, extended_encodings = BeamSearchOptimize(
            #     W_reshaped_sorted,
            #     self.codebooks[layer_name]['base'],
            #     self.codebooks[layer_name]['extended'],
            #     self.encodings[layer_name]['base'],
            #     self.encodings[layer_name]['extended']
            # )
            # self.encodings[layer_name] = {'base': B_base, 'extended': extended_encodings}
            # Now reconstruct W_encoded_reshaped_sorted from updated encodings and codebooks
            # This requires implementing the VQ reconstruction logic.


        # Step 9: Restore original channel order (rows)
        # Apply the inverse mapping to the rows of the sorted reshaped matrix
        W_encoded_reshaped_final = np.zeros_like(W_encoded_reshaped_sorted)
        # W_encoded_reshaped_final[channel_mapping] = W_encoded_reshaped_sorted
        # channel_mapping maps from original index to sorted index.
        # To restore original order, we need inverse mapping: sorted index -> original index
        inverse_channel_mapping = np.argsort(channel_mapping)
        W_encoded_reshaped_final[inverse_channel_mapping, :] = W_encoded_reshaped_sorted


        # Reshape back to original dimension if it was convolutional
        if is_conv:
            W_final = W_encoded_reshaped_final.reshape(original_shape)
        else:
            W_final = W_encoded_reshaped_final # Already 2D


        # Step 19: Block-level fine-tuning (optional)
        if model is not None and calibration_sample is not None:
             print("  Performing block-level fine-tuning...")
             # calibration_sample is (X_sample, y_sample)
             FineTuneBlock(model, block_idx, calibration_sample[0], calibration_sample[1], lr=1e-3, max_iter=10)


        # Calculate compression ratio based on original and compressed bit counts
        compression_ratio = self.calculate_compression_ratio(original_shape, W_reshaped.shape[1], n_important)


        quantization_info = {
            'importance': importance, # Importance of original rows (channels)
            'channel_mapping': channel_mapping, # Mapping from original row index to sorted row index
            'n_important': n_important,
            'compression_ratio': compression_ratio,
            'original_shape': original_shape,
            'reshaped_shape': W_reshaped.shape,
            'is_conv': is_conv,
            'padding_applied': padding,
            'num_vectors': vectors_sorted.shape[0] if vectors_sorted.size > 0 else 0,
            'codebook_size_base': C_base.shape[0] if C_base.size > 0 else 0,
            'codebook_size_extended': [c.shape[0] for c in self.codebooks[layer_name]['extended']]
        }

        print(f"  Quantization complete for {layer_name}. Compression ratio: {quantization_info['compression_ratio']:.2f}x")

        return W_final, quantization_info


    def quantize_model(self, model, calibration_data=None):
        """
        Quantize an entire model layer by layer.

        Args:
            model (nn.Module): PyTorch model to quantize
            calibration_data (DataLoader): Calibration data for computing activations

        Returns:
            nn.Module: Quantized model
        """
        print("=== Starting CRVQ Model Quantization ===")

        # Create a deep copy to avoid modifying the original model until quantization is final
        # Using state_dict is safer than deepcopy for PyTorch models
        original_state_dict = model.state_dict()
        quantized_model = type(model)() # Create a new instance of the same model class
        quantized_model.load_state_dict(original_state_dict)


        device = next(model.parameters()).device # Get model device
        quantized_model.to(device)
        quantized_model.eval() # Set to eval mode for activation collection

        # Get activations if calibration data provided
        activations = {}
        calibration_sample = None # Store a single sample for block fine-tuning
        if calibration_data is not None:
            activations = self.collect_activations(quantized_model, calibration_data)
            # Get one sample for block fine-tuning
            try:
                 data_iter = iter(calibration_data)
                 calibration_sample = next(data_iter)
                 calibration_sample = (calibration_sample[0].to(device), calibration_sample[1].to(device)) # Move to device
            except Exception as e:
                 print(f"Warning: Could not get calibration sample for block fine-tuning: {e}")
                 calibration_sample = None


        # Quantize each layer
        layer_count = 0
        quantization_infos = {}

        # Iterate through named modules to find layers to quantize
        for name, module in quantized_model.named_modules():
            # Check if the module is a quantizable layer (Linear or Conv) and has weights
            if isinstance(module, (nn.Linear, nn.Conv2d)) and hasattr(module, 'weight') and module.weight is not None:
                layer_count += 1
                print(f"\nProcessing layer {layer_count}: {name}")

                # Get weight matrix
                weight = module.weight.data # Use .data to avoid tracking gradients

                # Get corresponding activations (inputs to this layer)
                # We need the activation that is the input to the current module's weight multiplication.
                # The `collect_activations` hook is on `forward`, capturing input.
                # The key in `activations` dict is the module's name.
                layer_activations = activations.get(name, None)
                # Ensure activations are numpy and transposed to (N_flat x O) if needed
                if layer_activations is not None and torch.is_tensor(layer_activations):
                     layer_activations = layer_activations.detach().cpu().numpy()
                     # Simple heuristic for transposing if shape seems wrong for (N_flat x O)
                     # If first dim is batch size and larger than second, transpose
                     if layer_activations.shape[0] == calibration_data.batch_size and layer_activations.shape[0] > layer_activations.shape[1]:
                         layer_activations = layer_activations.T


                # Quantize layer
                quantized_weight, quant_info = self.quantize_layer(
                    weight,
                    layer_activations,
                    layer_name=name, # Use module name as layer name
                    block_idx=layer_count - 1, # 0-indexed block index
                    model=quantized_model, # Pass the model for block fine-tuning
                    calibration_sample=calibration_sample # Pass a calibration sample
                )

                # Update module weights (in-place or by assigning a new tensor)
                # Ensure quantized weight is a tensor and on the correct device
                module.weight.data = torch.tensor(quantized_weight, dtype=module.weight.dtype, device=device)

                # Store quantization info for this layer
                quantization_infos[name] = quant_info


        print(f"\n=== Model quantization complete. Quantized {layer_count} layers ===")

        # Step 21: End-to-end fine-tuning (optional)
        # E2E_FineTune(quantized_model, calibration_data, lr=1e-4, max_epochs=5)


        return quantized_model, quantization_infos


    def collect_activations(self, model, calibration_data, max_batches=10):
        """
        Collect activations from calibration data for computing Hessian proxy.
        Collects the *input* to the layer's `forward` method.

        Args:
            model (nn.Module): Model to collect activations from
            calibration_data (DataLoader): Calibration dataset
            max_batches (int): Maximum number of batches to process

        Returns:
            dict: Dictionary mapping layer names to concatenated activation matrices (input features x observations)
        """
        print("Collecting activations from calibration data...")

        activations = {}
        hooks = []
        device = next(model.parameters()).device


        def get_activation_hook(name):
            def hook(module, input, output):
                # Input is a tuple for some layers, get the main tensor
                if isinstance(input, tuple):
                    act = input[0]
                else:
                    act = input

                # Process activation tensor
                if torch.is_tensor(act):
                    # Ensure activation is on CPU and convert to numpy
                    act_np = act.detach().cpu().numpy()

                    # Reshape for linear/convolutional layers input
                    # For Linear (M x N), input is (batch x N)
                    # For Conv2d (O x I x kH x kW), input is (batch x I x H_in x W_in)
                    # We need to flatten the input features for matrix multiplication (N_flat x O)
                    # Flatten input features: (batch, I * H_in * W_in)
                    if len(act_np.shape) > 2: # Convolutional layer input
                         act_np = act_np.reshape(act_np.shape[0], -1) # Reshape to (batch, flattened_features)

                    # Transpose to (flattened_features x batch)
                    act_np = act_np.T

                    # Store
                    if name not in activations:
                        activations[name] = []
                    activations[name].append(act_np)
            return hook

        # Register hooks for layers whose weights will be quantized
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hook = module.register_forward_hook(get_activation_hook(name))
                hooks.append(hook)

        # Run forward passes to collect activations
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(calibration_data):
                if batch_idx >= max_batches:
                    break
                data = data.to(device) # Move data to the same device as the model
                _ = model(data)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Concatenate activations for each layer
        final_activations = {}
        for name, act_list in activations.items():
            if act_list:
                # Concatenate along the observations dimension (axis=1)
                final_activations[name] = np.concatenate(act_list, axis=1)

        print(f"Collected activations for {len(final_activations)} layers")
        # Print shapes of collected activations for debugging
        for name, act_matrix in final_activations.items():
            print(f"  Activation shape for {name}: {act_matrix.shape}")

        return final_activations

    def calculate_compression_ratio(self, original_shape, N_flat, n_important):
        """
        Calculate approximate compression ratio achieved by CRVQ.
        This is a simplified calculation.

        Args:
            original_shape (tuple): Shape of original weight matrix
            N_flat (int): Flattened input dimension (columns in reshaped 2D)
            n_important (int): Number of important channels (rows)

        Returns:
            float: Compression ratio
        """
        M = original_shape[0] # Number of output channels/rows
        original_params = np.prod(original_shape)
        original_bits = original_params * 32  # Assuming 32-bit floats

        # Calculate compressed size components:
        # Basic codebook: (2^e * d) * 32 bits
        # Extended codebooks: (m-1) * (2^e * d) * 32 bits
        # Basic encodings: (Total vectors) * e bits
        # Extended encodings: (Vectors in important part) * (m-1) * e bits

        codebook_size_base = (2**self.e) * self.d # Number of floats in base codebook
        codebook_size_extended = (self.m - 1) * (2**self.e) * self.d # Number of floats in all extended codebooks

        # Total number of vectors = (M * N_flat) / d
        total_vectors = (M * N_flat) // self.d # Integer division

        # Number of vectors in the important part = (n_important * N_flat) / d
        vectors_important = (n_important * N_flat) // self.d if n_important > 0 else 0

        codebook_bits = (codebook_size_base + codebook_size_extended) * 32
        encoding_bits_base = total_vectors * self.e
        encoding_bits_extended = vectors_important * (self.m - 1) * self.e

        compressed_bits = codebook_bits + encoding_bits_base + encoding_bits_extended


        # Avoid division by zero or very small numbers
        if compressed_bits <= 0:
            return float('inf') # Infinite compression if no bits used (shouldn't happen with codebooks)
        if original_bits <= 0:
            return 1.0 # No compression if original size is zero

        compression_ratio = original_bits / compressed_bits

        return compression_ratio


    def save_quantization_state(self, filepath):
        """Save quantization state to file."""
        state = {
            'codebooks': self.codebooks,
            'encodings': self.encodings,
            'channel_mappings': self.channel_mappings,
            'config': {
                'm': self.m,
                'lambda_ratio': self.lambda_ratio,
                'd': self.d,
                'e': self.e,
                'loss_threshold': self.loss_threshold
            }
        }
        np.savez_compressed(filepath, **state)
        print(f"Quantization state saved to {filepath}")

    def load_quantization_state(self, filepath):
        """Load quantization state from file."""
        state = np.load(filepath, allow_pickle=True)
        self.codebooks = state['codebooks'].item()
        self.encodings = state['encodings'].item()
        self.channel_mappings = state['channel_mappings'].item()

        config = state['config'].item()
        self.m = config['m']
        self.lambda_ratio = config['lambda_ratio']
        self.d = config['d']
        self.e = config['e']
        self.loss_threshold = config['loss_threshold']

        print(f"Quantization state loaded from {filepath}")
