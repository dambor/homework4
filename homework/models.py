from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden_dim: int = 128
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
            hidden_dim (int): dimension of hidden layers
        """
        super().__init__()
        
        # Input dimension: track_left and track_right concatenated
        # Each has shape (n_track, 2), so input_dim = n_track * 2 * 2
        input_dim = n_track * 2 * 2
        
        # Output dimension: future waypoints
        # Each waypoint has 2 coordinates (x, y)
        output_dim = n_waypoints * 2
        
        # Define the MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.n_track = n_track
        self.n_waypoints = n_waypoints

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        batch_size = track_left.shape[0]
        
        # Flatten the inputs
        track_left_flat = track_left.reshape(batch_size, -1)   # (b, n_track * 2)
        track_right_flat = track_right.reshape(batch_size, -1) # (b, n_track * 2)
        
        # Concatenate the inputs
        x = torch.cat([track_left_flat, track_right_flat], dim=1) # (b, n_track * 2 * 2)
        
        # Pass through MLP
        x = self.mlp(x)
        
        # Reshape to (b, n_waypoints, 2)
        x = x.reshape(batch_size, self.n_waypoints, 2)
        
        return x


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model
        
        # Learned query embeddings for each waypoint
        self.query_embed = nn.Embedding(n_waypoints, d_model)
        
        # Input projection for track points (left and right boundaries)
        self.track_encoder = nn.Sequential(
            nn.Linear(2, d_model),  # Project 2D coordinates to d_model
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Transformer decoder layer and full decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Output projection to get 2D waypoints
        self.output_proj = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        batch_size = track_left.shape[0]
        
        # Encode left and right track boundaries
        track_left_encoded = self.track_encoder(track_left)     # (b, n_track, d_model)
        track_right_encoded = self.track_encoder(track_right)   # (b, n_track, d_model)
        
        # Concatenate left and right track points
        track_encoded = torch.cat([track_left_encoded, track_right_encoded], dim=1)  # (b, 2*n_track, d_model)
        
        # Get query embeddings and expand to batch size
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)  # (b, n_waypoints, d_model)
        
        # Apply transformer decoder
        # - queries are the targets (what we want to predict)
        # - track_encoded is the memory (what we attend to)
        decoded = self.transformer_decoder(queries, track_encoded)  # (b, n_waypoints, d_model)
        
        # Project to 2D waypoints
        waypoints = self.output_proj(decoded)  # (b, n_waypoints, 2)
        
        return waypoints


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)
        
        # CNN backbone
        self.conv_layers = nn.Sequential(
            # First convolution block
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolution block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolution block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth convolution block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 4))  # Adaptive pooling to fixed size
        )
        
        # Calculate the flattened size after convolutions
        # For the input (3, 96, 128), after the above layers, we get (256, 3, 4)
        flattened_size = 256 * 3 * 4
        
        # Regression head to predict waypoints
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_waypoints * 2)  # Output: n_waypoints with (x, y) coordinates
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        batch_size = image.shape[0]
        
        # Normalize input
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        
        # Pass through CNN backbone
        features = self.conv_layers(x)
        
        # Pass through regressor to get waypoints
        waypoints_flat = self.regressor(features)
        
        # Reshape to (batch_size, n_waypoints, 2)
        waypoints = waypoints_flat.view(batch_size, self.n_waypoints, 2)
        
        return waypoints


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
