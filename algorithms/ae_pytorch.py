from math import sqrt
import numpy as np
import torch
from torch import device, no_grad, nn, manual_seed, tensor, float32
from torch.nn import Linear, Module, MSELoss, ReLU, Sequential, Sigmoid, Tanh
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam


class AutoEncoder(Module):
    '''A self-contained AutoEncoder that handles everything internally
    '''
    @staticmethod
    def build_layer(sizes, non_linearity=None):
        '''Construct encoder or decoder as a Sequential of Linear labels, with or without non-linearities

        Positional arguments:
            sizes   List of sizes for each Linear Layer
        Keyword arguments:
            non_linearity  Object used to introduce non-linearity between layers
        '''
        linears = [Linear(m, n) for m, n in zip(sizes[:-1], sizes[1:])]

        for id, layer in enumerate(linears):
            if id != len(linears) - 1:
                # nn.init.xavier_normal_(layer.weight, gain=0.5)
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            else:
                nn.init.xavier_normal_(layer.weight, gain=0.5)

        if non_linearity == None:
            return Sequential(*linears)
        else:
            return Sequential(*[item for pair in [(layer, non_linearity) if id != len(linears) - 1 else (layer, Tanh()) for id, layer in enumerate(linears)] for item in pair])

    def __init__(self, input_dim=None, hidden_dims=None, latent_dim=2, batch_size=128, encoder_non_linearity=None, decoder_non_linearity=None, epochs=25, lr=0.001):
        '''Initialize a self-contained autoencoder

        Keyword arguments:
            input_dim             Dimension of input data (will be inferred from data if not provided)
            hidden_dims           List of hidden layer dimensions (default: [128, 64, 32])
            latent_dim            Dimension of latent space (default: 2)
            batch_size            Batch size for training (default: 128)
            encoder_non_linearity Non-linearity for encoder (default: ReLU)
            decoder_non_linearity Non-linearity for decoder (default: ReLU)
        '''
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = [128, 64, 32] if hidden_dims is None else hidden_dims
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device("cpu")
        print(f"Device: {self.device}")

        # These will be set in fit_transform if not provided
        self.encoder_non_linearity = ReLU() if encoder_non_linearity is None else encoder_non_linearity
        self.decoder_non_linearity = ReLU() if decoder_non_linearity is None else decoder_non_linearity

        # These will be defined when input_dim is known (in fit_transform if not provided)
        self.encoder = None
        self.decoder = None

        self.encode = True
        self.decode = True

        # Build model if input_dim is provided
        if input_dim is not None:
            self._build_model()

    def _build_model(self):
        '''Build encoder and decoder networks'''
        # Define encoder and decoder sizes
        encoder_sizes = [self.input_dim] + self.hidden_dims + [self.latent_dim]
        decoder_sizes = [self.latent_dim] + self.hidden_dims[::-1] + [self.input_dim]

        # Create encoder and decoder
        self.encoder = AutoEncoder.build_layer(encoder_sizes, non_linearity=self.encoder_non_linearity)
        self.decoder = AutoEncoder.build_layer(decoder_sizes, non_linearity=self.decoder_non_linearity)

    def forward(self, x):
        '''Propagate value through network

           Computation is controlled by self.encode and self.decode
        '''
        if self.encode:
            x = self.encoder(x)

        if self.decode:
            x = self.decoder(x)
        return x

    def n_encoded(self):
        return self.latent_dim

    @staticmethod
    def spike_scaling_min_max(spikes, min_peak, max_peak):
        spikes_std = np.zeros_like(spikes)
        for col in range(len(spikes[0])):
            spikes_std[:, col] = (spikes[:, col] - min_peak) / (max_peak - min_peak)

        return spikes_std

    def fit_transform(self, X, y=None, seed=42, verbose=1):
        '''Train model on data and return latent codes

        Parameters:
            X          Input data as numpy array
            y          Labels (optional, just for tracking)
            epochs     Number of training epochs
            lr         Learning rate
            seed       Random seed for reproducibility
            device     Device to use (cpu or cuda)
            verbose    Verbosity level

        Returns:
            latent_codes    Numpy array of latent representations
        '''
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # X = self.spike_scaling_min_max(X, min_peak=np.amin(X), max_peak=np.amax(X)) * 2 - 1

        # Set input dimension if not already set
        if self.input_dim is None:
            self.input_dim = X.shape[1]
            self._build_model()

        # Move model to device
        dev = torch.device(self.device)
        self.to(dev)

        # Convert data to PyTorch tensors
        X_tensor = tensor(X, dtype=float32)
        if y is not None:
            y_tensor = tensor(y)
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            # Create dummy labels if none provided
            y_tensor = tensor(np.zeros(len(X)))
            dataset = TensorDataset(X_tensor, y_tensor)

        # Create data loader
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Create optimizer and loss function
        optimizer = Adam(self.parameters(), lr=self.lr)
        criterion = MSELoss()

        # Train the model
        self._train(data_loader, optimizer, criterion, epochs=self.epochs, verbose=verbose)

        # Get latent codes
        latent_codes = self._transform(X_tensor)

        return latent_codes

    def transform(self, X):
        '''Transform data to latent space without training

        Parameters:
            X           Input data as numpy array
            device      Device to use (cpu or cuda)

        Returns:
            latent_codes    Numpy array of latent representations
        '''
        X_tensor = tensor(X, dtype=float32)
        return self._transform(X_tensor, device=self.device)

    def _transform(self, X_tensor):
        '''Internal method to transform data to latent space

        Parameters:
            X_tensor    Input data as PyTorch tensor
            device      Device to use (cpu or cuda)

        Returns:
            latent_codes    Numpy array of latent representations
        '''
        # Move model to device if not already there
        dev = torch.device(self.device)
        self.to(dev)

        # Store original state
        save_decode = self.decode
        self.decode = False

        # Get latent codes
        X_tensor = X_tensor.to(dev)
        with no_grad():
            latent_codes = self(X_tensor).cpu().numpy()

        # Restore original state
        self.decode = save_decode

        return latent_codes

    def _train(self, data_loader, optimizer, criterion, epochs=25, verbose=1):
        '''Train the model

        Parameters:
            data_loader    DataLoader containing training data
            optimizer      Optimizer to use
            criterion      Loss function
            epochs         Number of training epochs
            device         Device to use (cpu or cuda)
            verbose        Verbosity level

        Returns:
            losses         List of training losses
        '''
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_features, _ in data_loader:
                batch_features = batch_features.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self(batch_features)
                loss = criterion(outputs, batch_features)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Record average loss for this epoch
            avg_loss = epoch_loss / len(data_loader)
            losses.append(avg_loss)

            if verbose == 1:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}')

        return losses


# Example usage
if __name__ == "__main__":
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # Generate random data
    X = np.random.rand(n_samples, n_features)

    # Create and train model
    model = AutoEncoder(
        hidden_dims=[70,60,50,40,30,20,10,5],
        latent_dim=2,
        batch_size=32
    )

    # Train and get latent codes in one step
    latent_codes = model.fit_transform(X, seed=42, verbose=1)

    print(f"\nLatent codes shape: {latent_codes.shape}")

    # Transform new data without retraining
    X_new = np.random.rand(10, n_features)
    new_latent_codes = model.transform(X_new)
    print(f"New latent codes shape: {new_latent_codes.shape}")