import torch
import torch.nn as nn 
import torchvision.models as models
    
class MyModel_MLP_transform(nn.Module):
    # input 5 inputs [delta_x, delta_y, delta_yaw, v_ref, v_ego] 
    # output: throttle, steering angle 
    def __init__(self, neurons): 
        super().__init__()
        self.neurons = neurons  
            
        self.predictor = nn.Sequential(
            nn.Linear(in_features = 5, out_features = self.neurons[0]),
            nn.BatchNorm1d(self.neurons[0]),
            nn.ReLU(), 
            nn.Linear(in_features = self.neurons[0], out_features = self.neurons[1]),
            nn.BatchNorm1d(self.neurons[1]),
            nn.ReLU(), 
            nn.Linear(in_features = self.neurons[1], out_features = self.neurons[2]),
            nn.BatchNorm1d(self.neurons[2]),
            nn.ReLU(), 
            nn.Linear(in_features = self.neurons[2], out_features = 2),  
        )  
          
    def forward(self, x):  
        return self.predictor(x)
    
    
    
class MyModel_CNN(nn.Module):
    # input: image, v_ref, v_ego
    # output: relative transformation, delta_x, delta_y, delta_yaw
    def __init__(self): 
        super().__init__()        
        self.features = models.alexnet(pretrained=True).features   
        self.extra = nn.MaxPool2d((2,2))
        self.predict = nn.Sequential(
            nn.Flatten(),                
            nn.Linear(258, 64),
            nn.ReLU(), 
            nn.Linear(64, 3),
        )

    def forward(self, x):
        img, ref_v, ego_v = x
        img = img.view(-1, 3, 128, 128).to(torch.float32)
        features = self.features(img)   
        features = self.extra(features) 
        features = features.view(-1, 256)
        features = torch.cat((features, ref_v.view(-1, 1), ego_v.view(-1, 1)), axis = 1).to(torch.float32) 
        preds = self.predict(features)
        return preds
        
    
 