import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc


# Example values
# Replace these later with real probabilities if needed

y_true = [0, 0, 1, 1]

y_scores = [0.1, 0.4, 0.35, 0.8]


# Compute ROC Curve
fpr, tpr, thresholds = roc_curve(

    y_true,

    y_scores
)

roc_auc = auc(

    fpr,

    tpr
)


# Plot ROC Curve
plt.figure(figsize=(8, 6))

plt.plot(

    fpr,

    tpr,

    label=f"AUC = {roc_auc:.2f}"
)

plt.plot(

    [0, 1],

    [0, 1],

    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.savefig(

    "outputs/plots/roc_curve.png"
)

plt.close()


print("ROC Curve saved successfully!")