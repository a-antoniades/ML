import numpy as np
import imageio
import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# np.set_printoptions(threshold=np.nan)

# Image to array function


def import_images(folder):
    images2 = np.empty([0, 3])
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            img = imageio.imread(os.path.join(folder, filename))
            img = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
            images2 = np.vstack((images2, img))

    return images2


# Creating dataframe

plastics = import_images("/Users/antonis/Desktop/Themas/DP1/plastics")
fish = import_images("/Users/antonis/Desktop/Themas/DP1/fish")

dfArray = np.vstack((plastics, fish))

dfPlastic = pd.DataFrame(plastics, columns=['1', '2', '3'])
dfFish = pd.DataFrame(fish, columns=['1', '2', '3'])

dfPlastic['4'] = 1
dfFish['4'] = 0

df = dfPlastic.append(dfFish, ignore_index=True)

# Running model

X = df[['1', '2', '3']]
Y = df['4']

X = sm.add_constant(X)  # add constant

model = sm.OLS(Y, X).fit()
predictions = model.predict(X)

print_model = model.summary()
print(print_model)

# Plotting fit

fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(model, 0, ax=ax)
ax.set_ylabel('4')
ax.set_xlabel('1')
ax.set_title('Linear Reg 1')
graph1 = plt.show()


