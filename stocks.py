TESTING=False #only do first 100
Q5 = True #weekly on mondays
data_size = 765 #size of good data
if Q5:
    data_size = 143
print("+++ SETTINGS +++\n Q5={}, TEST={}\n+++ ------- +++".format(Q5,TESTING))
import datetime
import os
import numpy as np
from csv import reader
import matplotlib.pyplot as plt
cols = "Date,Open,High,Low,Close,Volume,Adj Close"
data_path = "./finance_data/finance_data/data" #change depending on where data located

cwd = os.getcwd()
if "project 4" not in cwd:
    print("Error: Must be in project 4 directory")
    exit(1)

files = os.listdir(data_path)
#check to see if right dir.
print(files[:10], flush=True)

##### Get all the data from csv files and store in arrays
stock_names = []
stock_data = {}
for fname in files:
    #only take csv files
    #print("fname: " + fname)
    if "csv" in fname:
        stockname = fname[:-4]
        #Open the csv data file
        with open(data_path + "/" + fname) as f:
            csv_reader = reader(f)
            csv_data = []
            #first row is column field names
            first_row = True
            for row in csv_reader:
                if first_row:
                    first_row = False
                else:
                    if Q5: #only on mondays
                        if "-" in row[0]: #check date format
                            yyyy,mm,dd = row[0].split("-")
                        else:
                            mm,dd,yyyy = row[0].split("/")

                        day = datetime.datetime(int(yyyy), int(mm), int(dd))
                        if day.weekday() == 0: #monday
                            csv_data.append(float(row[4]))
                        
                    else:
                        csv_data.append(float(row[4])) #Only closing price
            #print("Closes: ", len(csv_data))
            #Make sure that all data has same length
            #print("len", len(csv_data))
            if len(csv_data) == data_size:
                stock_names.append(stockname)
                stock_data[stockname] = csv_data
                if TESTING:
                    if len(stock_names) >= 100:
                        break


#deal with Name sector.csv (can't find it??)
stock_sectors = {}
with open("./finance_data/finance_data/Name_sector.csv") as f:
    csv_reader = reader(f)
    csv_data = []
    #first row is column field names
    first_row = True
    for row in csv_reader:
        if first_row:
            first_row = False
        else:
            stock_sectors[row[0]] = row[1] 
   
#create color map
color_from_sector = {}
sectors = list(set(stock_sectors.values())) #unique
print("sectors: ", sectors, flush=True)
print(len(sectors)) #11
colors = ['red', 'blue', 'green', 'yellow', 'orange', 'black', 'purple', 'cyan', 'magenta', 'pink', 'teal']
for i in range(len(sectors)):
    color_from_sector[sectors[i]] = colors[i] 

colormap = []
for stock in stock_names:
    sector = stock_sectors[stock]
    color = color_from_sector[sector]
    colormap.append(color)

#print("=== check ===")
#for i in range(10):
#    print("stock: {}, sector: {}, color: {},{}".format(stock_names[i], stock_sectors[stock_names[i]], color_from_sector[stock_sectors[stock_names[i]]], colormap[i]))
#print("=== ... ===")


#Now convert closing prices to returns, and normalised returns
stock_returns = {}
stock_normalised_returns = {}
for stock in stock_data.keys():
    returns = []
    normalised_returns = []
    closing_prices = stock_data[stock]
    for i in range(len(closing_prices)-1):
        ret = closing_prices[i+1] / closing_prices[i] - 1 # ( pi+1 - pi ) / pi
        returns.append( ret )
        normalised_returns.append(np.log(1 + ret)) 

    stock_returns[stock] = returns
    stock_normalised_returns[stock] = normalised_returns
    

# get correlation data. Takes a while.
N_stocks = len(stock_names)
correlations = np.zeros((N_stocks,N_stocks)) # i => stock_names[i]
for i in range(N_stocks):
    if i % 5 == 0:
        print("i: ", i, flush=True)
    for j in range(i+1):
        #Check this(?)
        #print(stock_names[i], stock_names[j])
        #print(len(stock_normalised_returns[stock_names[i]]), len(stock_normalised_returns[stock_names[j]]))
        corr =  np.corrcoef(stock_normalised_returns[stock_names[i]], stock_normalised_returns[stock_names[j]])
        #print("corr: ", corr[0,1])
        correlations[i,j] = corr[0,1]
        correlations[j,i] = corr[0,1]

#print("CORR")
#print(correlations[:10,:10], flush=True)

#w_ij
edge_weights = np.sqrt(2-2*correlations)
#print("Corr vs edge weight size!", edge_weights.shape, np.array(correlations).shape)
edge_weights_flat = []
for i in range(N_stocks):
    for j in range(i+1):
        edge_weights_flat.append(edge_weights[i,j])

#print("wij")
#print(edge_weights[:10,:10], flush=True)


plt.hist(edge_weights_flat, bins=30)
title = "Histogram of edge weights"
if Q5:
    title = title + " (Weekly data)"
plt.title(title)
plt.savefig("results/" + title + ".png")
plt.show()

#get minimal spanning tree
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

print("Constructing MST")
MST = minimum_spanning_tree(csr_matrix(edge_weights))

#test
#print("MST")
#print(MST[:10], flush=True)
MST = MST.todense()

#graph
import networkx as nx

gr = nx.Graph()
#get edges
print("Getting edges")
edges = []
for i in range(N_stocks):
    for j in range(N_stocks):
        if MST[i,j] > 0:
            edges.append((i,j))
            gr.add_edge(i,j)
##gr.add_edges_from(edges) #edges
node_colormap = [colormap[node] for node in gr] #Since gr has nodes in wrong order :/
nx.draw(gr, node_size=10, node_color=node_colormap) #, labels=stock_names, with_labels=True)
title = "MST with nodes colored by sector"
if Q5:
    title = title + " (Weekly data)"
plt.title(title)
plt.savefig("results/" + title + ".png")
plt.show()

#prediction, case 1:
alpha = 0
for i in range(N_stocks):
    stock = stock_names[i]
    # |Qi| / |Ni| , Qi = set of neighbors in the same sector as node i. Ni is set of neighbors
    Ni = 0
    Qi = 0
    for j in range(N_stocks):
        #If j a neighbor:
        if MST[i,j] > 0 or MST[j,i] > 0:
            Ni += 1
            #If i and neighbor j in same sector:
            if stock_sectors[stock_names[i]] == stock_sectors[stock_names[j]]:
                Qi += 1
    #print("i, Ni, Qi", i, Ni, Qi)
    #print(MST[i,:])
    #print(MST[:,i])
    P = Qi/Ni
    alpha += P #P vi in Si

alpha = alpha /N_stocks

#prediction, case 2:
alpha2 = 0
for i in range(N_stocks):
    stock = stock_names[i]
    # |Si| / |V| , Si = sector i, V = graph
    Si = 0
    V = len(stock_sectors) 
    #check how many stocks in same sector
    for val in stock_sectors.values():
        if (stock_sectors[stock_names[i]] == val):
            Si += 1

    P = Si/V
    alpha2 += P #P vi in Si

alpha2 = alpha2 /N_stocks
print("alpha 1 vs 2:", alpha, alpha2)