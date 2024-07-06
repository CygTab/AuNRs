from torch import nn


class MultiOutputRegressor1l(nn.Module):
    def __init__(self, input_dim, output_dim, hid1):
        super(MultiOutputRegressor1l, self).__init__()
        self.layer1 = nn.Linear(input_dim, hid1)
        self.output_layer = nn.Linear(hid1, output_dim)
        self.act = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.act(self.layer1(x))
        x = self.output_layer(x)
        return x


class MultiOutputRegressor3l(nn.Module):
    def __init__(self, input_dim, output_dim, hid1, hid2, hid3):
        super(MultiOutputRegressor3l, self).__init__()
        self.layer1 = nn.Linear(input_dim, hid1)
        self.layer2 = nn.Linear(hid1, hid2)
        self.layer3 = nn.Linear(hid2, hid3)
        self.output_layer = nn.Linear(hid3, output_dim)
        self.act = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        x = self.act(self.layer3(x))
        x = self.output_layer(x)
        return x


class MultiOutputRegressor5l(nn.Module):
    def __init__(self, input_dim, output_dim, hid1, hid2, hid3, hid4, hid5):
        super(MultiOutputRegressor5l, self).__init__()
        self.layer1 = nn.Linear(input_dim, hid1)
        self.layer2 = nn.Linear(hid1, hid2)
        self.layer3 = nn.Linear(hid2, hid3)
        self.layer4 = nn.Linear(hid3, hid4)
        self.layer5 = nn.Linear(hid4, hid5)
        self.output_layer = nn.Linear(hid5, output_dim)
        self.act = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        x = self.act(self.layer3(x))
        x = self.act(self.layer4(x))
        x = self.act(self.layer5(x))
        x = self.output_layer(x)
        return x