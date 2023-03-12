import torch
from sklearn.preprocessing import QuantileTransformer

class Scaler():
    def __init__(self, x: torch.FloatTensor) -> None:
        with torch.no_grad():
            x = x.clone()
            x[..., 0] = torch.log(x[..., 0] + 1.)
            # For feature 0
            #self.quantile_transformer_0 = QuantileTransformer(
            #    n_quantiles=500, output_distribution="normal", random_state=42
            #)
            #feature_shape = x[..., 0].shape
            #raveled_feature = x[..., 0].cpu().numpy().reshape(-1, 1)
            #self.quantile_transformer_0.fit(raveled_feature)
            #quantile_transformed = self.quantile_transformer_0.transform(raveled_feature)
            #x[..., 0] = torch.tensor(
            #    quantile_transformed.reshape(feature_shape),
            #    device=x.device)
            # End for feature 0
            # For feature 1
            self.quantile_transformer_1 = QuantileTransformer(
                n_quantiles=500, output_distribution="normal", random_state=42
            )
            feature_shape = x[..., 1].shape
            raveled_feature = x[..., 1].cpu().numpy().reshape(-1, 1)
            self.quantile_transformer_1.fit(raveled_feature)
            quantile_transformed = self.quantile_transformer_1.transform(raveled_feature)
            x[..., 1] = torch.tensor(
                quantile_transformed.reshape(feature_shape),
                device=x.device)
            # End for feature 1
            #self.max = torch.stack([torch.max(x[..., i]) for i in range(x.shape[-1])])
            #self.min = torch.stack([torch.max(x[..., i]) for i in range(x.shape[-1])])
            self.mean = torch.mean(x, dim=(-3, -2))
            self.std = torch.std(x, dim=(-3, -2))

    def scale(self, x: torch.FloatTensor):
        with torch.no_grad():
            x = x.clone()
            #min = self.min.to(x.device)
            #max = self.max.to(x.device)
            mean = self.mean.to(x.device)
            std = self.std.to(x.device)
            x[..., 0] = torch.log(x[..., 0] + 1.)
            #print(x[..., 1])
            
            # For feature 0
            #feature_shape = x[..., 0].shape
            #raveled_feature = x[..., 0].cpu().numpy().reshape(-1, 1)
            #quantile_transformed = self.quantile_transformer_0.transform(raveled_feature)
            #x[..., 0] = torch.tensor(
            #    quantile_transformed.reshape(feature_shape),
            #    device=x.device)
            # End for feature 0
            # For feature 1
            feature_shape = x[..., 1].shape
            raveled_feature = x[..., 1].cpu().numpy().reshape(-1, 1)
            quantile_transformed = self.quantile_transformer_1.transform(raveled_feature)
            x[..., 1] = torch.tensor(
                quantile_transformed.reshape(feature_shape),
                device=x.device)
            # End for feature 1
            #x = (x - min) / (max - min)
            x = (x - mean) / std
            return x

    def un_scale(self, x: torch.FloatTensor):
        with torch.no_grad():
            x = x.clone()
            #min = self.min.to(x.device)
            #max = self.max.to(x.device)
            mean = self.mean.to(x.device)
            std = self.std.to(x.device)
            x = x * std + mean
            #x = x * (max - min) + min
            x[..., 0] = torch.exp(x[..., 0]) - 1.
            
            # For feature 0
            #feature_shape = x[..., 0].shape
            #raveled_feature = x[..., 0].cpu().numpy().reshape(-1, 1)
            #quantile_transformed = self.quantile_transformer_0.inverse_transform(raveled_feature)
            #x[..., 0] = torch.tensor(
            #    quantile_transformed.reshape(feature_shape),
            #    device=x.device)
            # End for feature 0
            # For feature 1
            feature_shape = x[..., 1].shape
            raveled_feature = x[..., 1].cpu().numpy().reshape(-1, 1)
            quantile_transformed = self.quantile_transformer_1.inverse_transform(raveled_feature)
            x[..., 1] = torch.tensor(
                quantile_transformed.reshape(feature_shape),
                device=x.device)
            # End for feature 1
            return x