import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


# Confusion matrix values
cm = np.array([

    [4924, 969],

    [264, 6980]
])


# Labels
labels = [

    "Fake",

    "Real"
]


# Create display
disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=labels
)


# Plot
fig, ax = plt.subplots(figsize=(7, 7))

disp.plot(ax=ax)

plt.title("Confusion Matrix")

plt.savefig(
    "outputs/plots/confusion_matrix.png"
)

plt.show()

print("Confusion matrix saved successfully!")