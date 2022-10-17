import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

class ResNetBase(nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 32 
    PLANES = (16,32,64,64)

    def __init__(self, in_channels, out_channels, D=3):
        nn.Module.__init__(self)
        self.D = D

        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):
        # Sparse Matrix Query 
        self.inplanes = self.INIT_DIM
        self.layer1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=5, stride=1, dimension=D),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))
        self.layer3 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))           
        self.layer4 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))
        self.layer5 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True)) 
        self.layer6 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))
        self.layer7 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))  
        self.layer8 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))
        self.layer9 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))  
        self.layer10 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))
        self.layer11 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))  
        self.layer12 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))
        self.layer13 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))  
        self.layer14 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True))

        self.glob_pool = nn.Sequential(
            ME.MinkowskiGlobalAvgPooling(),
            ME.MinkowskiToFeature())

        self.feature = nn.Sequential(
          nn.Linear(3, 64),
          nn.ReLU(),
          nn.Linear(64,32),
        )
        
        self.matrix_embedding = nn.Sequential(
          nn.Linear(self.INIT_DIM*14+32, 256),
          nn.ReLU(),
          nn.Linear(256,128),
        )

        # Super Schedule
        self.isplit = nn.Embedding(17, 32)
        self.ksplit = nn.Embedding(17, 32)
        self.jsplit = nn.Embedding(8, 32)
        self.formati1 = nn.Embedding(2, 32)
        self.formati0 = nn.Embedding(2, 32)
        self.formatk1 = nn.Embedding(2, 32)
        self.formatk0 = nn.Embedding(2, 32)
        self.parchunk = nn.Embedding(9, 32) # For OpenTuner
        self.order = nn.Linear(36, 32) #6x6 Permutation

        self.schedule_embedding = nn.Sequential(
            nn.Linear(32*9,128),
            nn.ReLU(),
            nn.Linear(128,128),
        )
        
        # Final Layer
        self.final = nn.Sequential(
            nn.Linear(128+128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        );

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
   
    def embed_sparse_matrix(self, x1: ME.SparseTensor, x2) :
        # Sparse Matrix
        y1 = self.layer1(x1)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        y4 = self.layer4(y3)
        y5 = self.layer5(y4)
        y6 = self.layer6(y5)
        y7 = self.layer7(y6)
        y8 = self.layer8(y7)
        y9 = self.layer9(y8)
        y10 = self.layer10(y9)
        y11 = self.layer11(y10)
        y12 = self.layer12(y11)
        y13 = self.layer13(y12)
        y14 = self.layer14(y13)


        y1  = self.glob_pool(y1)
        y2  = self.glob_pool(y2)
        y3  = self.glob_pool(y3)
        y4  = self.glob_pool(y4)
        y5  = self.glob_pool(y5)
        y6  = self.glob_pool(y6)
        y7  = self.glob_pool(y7)
        y8  = self.glob_pool(y8)
        y9  = self.glob_pool(y9)
        y10 = self.glob_pool(y10)
        y11 = self.glob_pool(y11)
        y12 = self.glob_pool(y12)
        y13 = self.glob_pool(y13)
        y14 = self.glob_pool(y14)
        
        #y = F.normalize(torch.cat((y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14), dim=1))
        y = torch.cat((y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14), dim=1)

        x2 = self.feature(x2[:, :3])
        x1x2 = torch.cat((y,x2), dim=1)
        x1x2 = self.matrix_embedding(x1x2)
        
        #x1x2 = F.normalize(x1x2)

        return x1x2

    def embed_super_schedule(self, y) :
        # Super Schedule
        isplit = self.isplit(y[:, 0].long())
        ksplit = self.ksplit(y[:, 1].long())
        jsplit = self.jsplit(y[:, 2].long())
        order = self.order(y[:, 3:39])
        i1f = self.formati1(y[:, 39].long())
        i0f = self.formati0(y[:, 40].long())
        k1f = self.formatk1(y[:, 41].long())
        k0f = self.formatk0(y[:, 42].long())
        #pidx = self.paridx(y[:, 43].long())
        #pnum = self.parnum(y[:, 44].long())
        pchk = self.parchunk(y[:, 45].long())
        y = torch.cat((isplit,ksplit,jsplit,order,i1f,i0f,k1f,k0f,pchk), dim=1)
        y = self.schedule_embedding(y)

        #y = F.normalize(y)
        return y

    def forward_after_query(self, x, y):
        y = self.embed_super_schedule(y)
        xy = torch.cat((x,y), dim=1)
        xy = self.final(xy)
        return xy
    
    def forward(self, x: ME.SparseTensor, x2, y):
        # Concat - Final
        x = self.embed_sparse_matrix(x1,x2)
        y = self.embed_super_schedule(y)
        xy = torch.cat((x,y), dim=1)
        xy = self.final(xy)
        return xy


class ResNet14(ResNetBase):
    LAYERS = (1, 1, 1, 1)

