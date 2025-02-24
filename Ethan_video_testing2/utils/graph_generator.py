import matplotlib.pyplot as plt

def generate_graph(data, output_path):
    time = [entry['time'] for entry in data]
    mean_intensity = [entry['mean_intensity'] for entry in data]

    plt.figure()
    plt.plot(time, mean_intensity, label='Mean Intensity')
    plt.xlabel('Time')
    plt.ylabel('Mean Intensity')
    plt.title('Mean Intensity Over Time')
    plt.legend()
    plt.savefig(output_path)
    plt.show()
    plt.close()

def plot_fitted_curve(time, mean_intensity, model, popt, output_path):
    fitted_intensity = model(time, *popt)

    plt.figure()
    plt.plot(time, mean_intensity, 'b-', label='Data')
    plt.plot(time, fitted_intensity, 'r-', label='Fitted Curve')
    plt.xlabel('Time')
    plt.ylabel('Mean Intensity')
    plt.title('Mean Intensity with Fitted Curve')
    plt.legend()
    plt.savefig(output_path)
    plt.show()
    plt.close()

def generate_rate_of_change_graph(data, output_path):
    time = [entry['time'] for entry in data]
    flow_magnitude = [entry['flow_magnitude'] for entry in data]

    # Calculate the rate of change (derivative) of flow magnitude
    rate_of_change = [(flow_magnitude[i+1] - flow_magnitude[i]) / (time[i+1] - time[i]) for i in range(len(flow_magnitude) - 1)]
    time = time[:-1]  # Adjust time array to match the length of rate_of_change

    # Print the rate of change data for debugging
    print(f"Rate of change data: {rate_of_change}")

    plt.figure()
    plt.plot(time, rate_of_change, label='Rate of Change of Flow Magnitude')
    plt.xlabel('Time')
    plt.ylabel('Rate of Change')
    plt.title('Rate of Change of Flow Magnitude Over Time')
    plt.legend()
    plt.savefig(output_path)
    plt.show()
    plt.close()