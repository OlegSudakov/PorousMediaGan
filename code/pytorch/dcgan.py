import torch
import torch.nn as nn
import torch.nn.parallel

class DCGAN3D_D(nn.Container):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN3D_D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        # input is nc x isize x isize
        self.input_conv = nn.Conv3d(nc, ndf//2, 4, 2, 1, bias=False)
        self.input_lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.mm_conv = nn.Conv3d(4, ndf//2, 4, 2, 1, bias=False)
        self.mm_lrelu = nn.LeakyReLU(0.2, inplace=True)
        main = nn.Sequential(
        )
        i, csize, cndf = 3, isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(str(i),
                            nn.Conv3d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module(str(i+1),
                            nn.BatchNorm3d(cndf))
            main.add_module(str(i+2),
                            nn.LeakyReLU(0.2, inplace=True))
            i += 3

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module(str(i),
                            nn.Conv3d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module(str(i+1),
                            nn.BatchNorm3d(out_feat))
            main.add_module(str(i+2),
                            nn.LeakyReLU(0.2, inplace=True))
            i+=3
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4 x 4
        main.add_module(str(i),
                        nn.Conv3d(cndf, 1, 4, 1, 0, bias=False))
        main.add_module(str(i+1), nn.Sigmoid())
        
        self.main = main


    def forward(self, input, mm):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        input = self.input_lrelu(self.input_conv(input))
        mm = self.mm_lrelu(self.mm_conv(mm))
        input = torch.cat([input, mm], 1)
        output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        return output.view(-1, 1)

class DCGAN3D_G(nn.Container):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN3D_G, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2
        # input is Z, going into a convolution
        self.input_tconv = nn.ConvTranspose3d(nz, cngf//2, 4, 1, 0, bias=False)
        self.input_bnorm = nn.BatchNorm3d(cngf)
        self.input_relu = nn.ReLU(True)
        self.mm_tconv = nn.ConvTranspose3d(4, cngf//2, 4, 1, 0, bias=False)
        self.mm_bnorm = nn.BatchNorm3d(cngf)
        self.mm_relu = nn.ReLU(True)
        main = nn.Sequential(
        )

        i, csize, cndf = 3, 4, cngf
        while csize < isize//2:
            main.add_module(str(i),
                nn.ConvTranspose3d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module(str(i+1),
                            nn.BatchNorm3d(cngf//2))
            main.add_module(str(i+2),
                            nn.ReLU(True))
            i += 3
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(str(i),
                            nn.Conv3d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module(str(i+1),
                            nn.BatchNorm3d(cngf))
            main.add_module(str(i+2),
                            nn.ReLU(True))
            i += 3

        main.add_module(str(i),
                        nn.ConvTranspose3d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module(str(i+1), nn.Tanh())
        self.main = main

    def forward(self, input, mm):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        input = self.input_relu(self.input_bnorm(self.input_tconv(input)))
        mm = self.mm_relu(self.mm_bnorm(self.mm_tconv(mm)))
        input = torch.cat([input, mm], 1)
        return nn.parallel.data_parallel(self.main, input, gpu_ids)

class DCGAN3D_G_CPU(nn.Container):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN3D_G_CPU, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2
        # input is Z, going into a convolution
        self.input_tconv = nn.ConvTranspose3d(nz, cngf, 4, 1, 0, bias=False)
        self.input_bnorm = nn.BatchNorm3d(cngf)
        self.input_relu = nn.ReLU(True)
        self.mm_tconv = nn.ConvTranspose3d(4, cngf, 4, 1, 0, bias=False)
        self.mm_bnorm = nn.BatchNorm3d(cngf)
        self.mm_relu = nn.ReLU(True)
        main = nn.Sequential(
        )

        i, csize, cndf = 3, 4, cngf
        while csize < isize//2:
            main.add_module(str(i),
                nn.ConvTranspose3d(cngf, cngf//2, 4, 2, 1, bias=True))
            main.add_module(str(i+1),
                            nn.BatchNorm3d(cngf//2))
            main.add_module(str(i+2),
                            nn.ReLU(True))
            i += 3
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(str(i),
                            nn.Conv3d(cngf, cngf, 3, 1, 1, bias=True))
            main.add_module(str(i+1),
                            nn.BatchNorm3d(cngf))
            main.add_module(str(i+2),
                            nn.ReLU(True))
            i += 3

        main.add_module(str(i),
                        nn.ConvTranspose3d(cngf, nc, 4, 2, 1, bias=True))
        main.add_module(str(i+1), nn.Tanh())
        self.main = main

    def forward(self, input, mm):
        input = self.input_relu(self.input_bnorm(self.input_tconv(input)))
        mm = self.mm_relu(self.mm_bnorm(self.mm_tconv(mm)))
        input = torch.cat([input, mm], 1)
        return self.main(input)