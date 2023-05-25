import numpy as np
from sklearn.model_selection import train_test_split
import os
from lib.utils import *
# from lib.agents import *

#label2id = {tool: idx-1 for idx, tool in enumerate(os.listdir(dataset_folder)) if tool[0] != '.'}
#id2label = {v: k for k, v in label2id.items()}

POSITIVE_TOOL = 'rope'
HEIGHT = 4
WIDTH = 4
SIZE = (HEIGHT, WIDTH)
SEED = 25
#N_AGENTS = 1

np.random.seed(SEED)

dataset_folder = os.path.join(os.getcwd(), 'mechanical_tools')
tools = [tool for tool in os.listdir(dataset_folder) if tool[0] != '.']

df = get_data(dataset_folder, tools, POSITIVE_TOOL, SIZE, samples=200, balanced=True)
df.to_pickle('dataset.pkl')

#df_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED, shuffle=True)
#print(f'Total samples:\t{len(df)}')
#print(f'Train samples:\t{len(df_train)}')
#print(f' Test samples:\t{len(df_test)}\n')

print(f"Num positives:\t{np.sum(df['label'] == 1)}")
print(f"Num negatives:\t{np.sum(df['label'] != 1)}\n")

#samples_agent = len(df_train) // N_AGENTS + 1
#data = {n: df_train.iloc[n::N_AGENTS, :] for n in range(N_AGENTS)}

#for n in range(N_AGENTS):
#    print(f'Agent {n} will use {len(data[n])} samples.')

#da0 = DistributedAgents(5, data, batch_size=1)
#da1 = DistributedAgents(5, data, batch_size=1)
#print(*da0.net.parameters())

