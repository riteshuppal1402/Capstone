import pandas as pd
import pickle
import mpld3
import torch
import os
import numpy as np
import matplotlib.patches as mpatches
# matplotlib.use('agg')
import matplotlib
matplotlib.use("TkAgg")
#matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch import optim
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from flask import render_template, url_for



application = Flask(__name__)
cors = CORS(application)
x_output = np. load(r"file.npy")
#print(len(x_output[0][0][0][0]))
@application.route("/")
def index():
    return render_template("index.html")

def plot_results(y_preds):
  
    d = '2021-3-14 06:00'
    x = pd.date_range(d, periods=16, freq='60min')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.plot(x, y_true, label='True Data')
    ax.plot(x, y_preds, label='Predicted',color='blue')
    # ax.plot(x, y_preds, label='High Flow',color='red')

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')
    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    # figure(figsize=(180, 160), dpi=180)
    
    threshold = 60
    below_threshold = y_preds < threshold
    plt.scatter(x[below_threshold], y_preds[below_threshold], color='green') 
    above_threshold = np.logical_not(below_threshold)
    plt.scatter(x[above_threshold], y_preds[above_threshold], color='red')
    red_patch = mpatches.Patch(color='red', label='High Congestion')
    blue_patch = mpatches.Patch(color='green', label='Low Congestion')

    plt.legend(handles=[red_patch, blue_patch])
    
    fig1=plt.gcf()
    html_str1=mpld3.fig_to_html(fig1)
    return html_str1
    # plt.close()
class PCEModule(torch.nn.Module):
    def __init__(self, n_channels=10, n_classes=192, dropout_probability=0.2):
        super(PCEModule, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dropout_probability = dropout_probability
        # Layers ----------------------------------------------    
		 # Convolution Branch Layers
        self.all_conv_low = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, padding=1),
            torch.nn.ReLU(),
          #  torch.nn.AvgPool2d(2),
            # torch.nn.Conv2d(in_channels=4, out_channels=12, kernel_size=3, padding=1),
            # torch.nn.ReLU(),
          #  torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(in_channels=12, out_channels=8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_probability),
            # torch.nn.AvgPool2d(2)
        ) for joint in range(n_channels)])    
		#  Fully Connected Layers
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features= (96* (self.n_channels)), out_features=496), 
            torch.nn.ReLU(),
            # torch.nn.Linear(in_features= (14976), out_features=7000), 
            # torch.nn.ReLU(),
            # torch.nn.Linear(in_features= (14970), out_features=2000), 
            # torch.nn.ReLU(),
            torch.nn.Linear(in_features=496, out_features=n_classes)
        )
        # Initialization --------------------------------------
        for layer in self.fc:
           if layer.__class__.__name__ == "Linear":
               torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('relu'))
               torch.nn.init.constant_(layer.bias, 0.1)

    def forward(self, input):
        """
        This function performs the actual computations of the network for a forward pass.
        """
        # print(input.shape)
        # inputs_to_convolve = input[:,:,0:self.n_channels];
        # inputs_to_fc = input[:,0,0:self.n_channels];
        all_features = []
        # print(len(input[0]))
        for channel in range(0, self.n_channels):
           input_channel = input[:,channel]
          #  print(input_channel.shape)
           #input_channel = input_channel.unsqueeze(1)
           low = self.all_conv_low[channel](input_channel)
           output_channel = torch.cat([low], dim=1)
           all_features.append(output_channel)
        all_features = torch.cat(all_features, dim=1)
        # print(len(all_features))
        all_features = all_features.view(-1, 96 * (self.n_channels))  
        # all_features = torch.cat((all_features), dim=1);
        # all_features = torch.cat((all_features,inputs_to_fc), dim=1);
        output = self.fc(all_features)        
        return output
@application.route("/", methods=["GET","POST"])
def normal():
    if request.method == 'POST':
        data=request.form.get('date')
        
        string_input_with_date = data
        past = datetime.strptime(string_input_with_date, "%Y-%m-%d")
        present = datetime.now()
        if past.date() > present.date():
            return "Please select date before today's date ie "+str(datetime.now().strftime('%Y-%m-%d'))
        day=int(data[-2:])
        
        model=pickle.load(open('model.pkl','rb'))
        output=model(torch.Tensor(x_output))
        val=output[67-18+day][::12].detach().numpy()
        ans=plot_results(val)
        return render_template('index.html', ans=ans)
        

    return "Yippie"





if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    application.run(host='0.0.0.0', port=port, debug=True)
    # application.run(debug=True)
    # application.run(host="0.0.0.0", port=80)