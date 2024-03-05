from matplotlib import pyplot as plt


def visualize_metrics(skill_net, epoch_list, acc_list, auc_list, rmse_list):
    plt.figure(figsize=(10, 6))
    # Plot AUC
    plt.subplot(3, 1, 1)
    plt.plot(epoch_list, auc_list, label='AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC Over Epochs')
    plt.grid(True)
    # Plot ACC
    plt.subplot(3, 1, 2)
    plt.plot(epoch_list, acc_list, label='ACC')
    plt.xlabel('Epoch')
    plt.ylabel('ACC')
    plt.title('ACC Over Epochs')
    plt.grid(True)
    # Plot RMSE
    plt.subplot(3, 1, 3)
    plt.plot(epoch_list, rmse_list, label='RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('RMSE Over Epochs')
    plt.grid(True)

    plt.tight_layout()  # Adjust spacing between subplots

    # Save the plot
    plt.savefig(f"plots/model_performance_{skill_net.tag}.png")  # Customize filename
