# import matplotlib.pyplot as plt

# # Data for Yahoo!Movies
# weights = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# mae_yahoo = {
#     r"$\alpha = 0.1, \beta = 0.1$":[0.6404, 0.6693, 0.6482, 0.6316, 0.6475, 0.6574, 0.6311, 0.648],
#     r"$\alpha = 0.5, \beta = 0.5$":[0.6211, 0.6188, 0.6155, 0.6109, 0.6166, 0.6215, 0.6232, 0.6208],
#     r"$\alpha = 0.5, \beta = 0.1$":[0.6344, 0.6488, 0.6475, 0.6523, 0.6655, 0.6699, 0.655, 0.6721],
#     r"$\alpha = 0.1, \beta = 0.5$":[0.6444, 0.6388, 0.651, 0.658, 0.659, 0.6501, 0.666, 0.6722]
# }

# # Data for BeerAdvocate
# mae_beer = {
#     r"$\alpha = 0.1, \beta = 0.1$":[0.4305, 0.4533, 0.435, 0.4404, 0.4199, 0.4317, 0.4486, 0.428],
#     r"$\alpha = 0.5, \beta = 0.5$":[0.4283, 0.4255, 0.4226, 0.4199, 0.4216, 0.4288, 0.4305, 0.4321],
#     r"$\alpha = 0.5, \beta = 0.1$":[0.4251, 0.4289, 0.4318, 0.4388, 0.445, 0.448, 0.422, 0.4511],
#     r"$\alpha = 0.1, \beta = 0.5$":[0.4316, 0.4366, 0.4301, 0.4134, 0.4489, 0.4598, 0.4515, 0.4611]
# }


# # Font size variable
# font_size = 18

# col = ["red", "green", "blue", "purple"]

# ## TODO: Explain in the caption: reference the equations.

# for dataset in [mae_yahoo, mae_beer]:
#     plt.figure(figsize=(14, 8))
#     for i, (label, y) in enumerate(dataset.items()):
#         plt.plot([x * .1 for x in weights], y, marker='o', linestyle='-', color=col[i], label=label, markersize=10, linewidth=2)
#     plt.xlabel(r'$\lambda$', fontsize=font_size)
#     plt.ylabel('MAE', fontsize=font_size)
#     plt.legend(fontsize=font_size)
#     plt.xticks(fontsize=font_size)
#     plt.yticks(fontsize=font_size)
#     plt.grid(True, linestyle='--', linewidth=0.5)
# plt.show()


import matplotlib.pyplot as plt

# Data for Yahoo!Movies
weights = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

mae_yahoo = {
    r"$\alpha = 0.1, \beta = 0.1$":[0.6404, 0.6693, 0.6482, 0.6316, 0.6475, 0.6574, 0.6311, 0.648],
    r"$\alpha = 0.5, \beta = 0.5$":[0.6211, 0.6188, 0.6155, 0.6109, 0.6166, 0.6215, 0.6232, 0.6208],
    r"$\alpha = 0.5, \beta = 0.1$":[0.6344, 0.6488, 0.6475, 0.6523, 0.6655, 0.6699, 0.655, 0.6721],
    r"$\alpha = 0.1, \beta = 0.5$":[0.6444, 0.6388, 0.651, 0.658, 0.659, 0.6501, 0.666, 0.6722]
}

# Data for BeerAdvocate
mae_beer = {
    r"$\alpha = 0.1, \beta = 0.1$":[0.4305, 0.4533, 0.435, 0.4404, 0.4199, 0.4317, 0.4486, 0.428],
    r"$\alpha = 0.5, \beta = 0.5$":[0.4283, 0.4255, 0.4226, 0.4199, 0.4216, 0.4288, 0.4305, 0.4321],
    r"$\alpha = 0.5, \beta = 0.1$":[0.4251, 0.4289, 0.4318, 0.4388, 0.445, 0.448, 0.422, 0.4511],
    r"$\alpha = 0.1, \beta = 0.5$":[0.4316, 0.4366, 0.4301, 0.4134, 0.4489, 0.4598, 0.4515, 0.4611]
}

# Font size variable
font_size = 18

col = ["red", "green", "blue", "purple"]

# Dataset names for reference
datasets = {
    'Yahoo!Movies': mae_yahoo,
    'BeerAdvocate': mae_beer
}

for dataset_name, dataset in datasets.items():
    plt.figure(figsize=(14, 8))
    for i, (label, y) in enumerate(dataset.items()):
        plt.plot([x * .1 for x in weights], y, marker='o', linestyle='-', color=col[i], label=label, markersize=10, linewidth=2)
    plt.title(f'{dataset_name} Dataset', fontsize=font_size)  # Add title with dataset name
    plt.xlabel(r'$\lambda$', fontsize=font_size)
    plt.ylabel('MAE', fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()
