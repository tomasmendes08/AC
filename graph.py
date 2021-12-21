import pandas as pd
from collections import Counter
import seaborn as sb
import matplotlib.pyplot as plt

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

sb.heatmap(train, annot=True)
plt.show()

# ages = list(train['owner_age'])

# ages = pd.DataFrame(ages, columns=["Age"])

# plt.hist(ages['Age'], bins=10, edgecolor='black')
# plt.show()

#sb.set(rc={'figure.figsize':(11.7,8.27)})
# sb.set(style="darkgrid")
# sb

# sb.histplot(data=ages, x="Age", bins=10).set_title("Owner age at the time of loan")
# plt.ylabel("Frequency")
# plt.show()

# df = pd.DataFrame.from_dict(count, orient='index')

# # plt.xlabel("Age", fontdict={'fontsize': 80, 'fontweight': 'bold'})
# # plt.ylabel("Frequency", fontdict={'fontsize': 80, 'fontweight': 'bold'})

# df.plot(kind='bar')

# fig = plt.figure(figsize=(16,9))
# ax = fig.add_axes([0,0,1,1])
# plt.title('Age frequency', fontsize=20)
# ax.bar(genres_dict_sorted.keys(),genres_dict_sorted.values(),
# # plt.xlabel('Genres', fontsize=16)
# # plt.ylabel('Number of Votes', fontsize=16)
# plt.show()