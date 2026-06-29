import csv
import matplotlib.pyplot as plt

steps = []
true_x = []
true_y = []
est_x = []
est_y = []
position_error = []
n_eff = []

with open("fastslam_results.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        steps.append(int(row["step"]))
        true_x.append(float(row["true_x"]))
        true_y.append(float(row["true_y"]))
        est_x.append(float(row["est_x"]))
        est_y.append(float(row["est_y"]))
        position_error.append(float(row["position_error"]))
        n_eff.append(float(row["effective_sample_size"]))

plt.figure()
plt.plot(true_x, true_y, label="True trajectory")
plt.plot(est_x, est_y, label="Estimated trajectory")
plt.xlabel("x")
plt.ylabel("y")
plt.title("FastSLAM Trajectory")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.savefig("trajectory_plot.png")
plt.show()

plt.figure()
plt.plot(steps, position_error)
plt.xlabel("Step")
plt.ylabel("Position Error")
plt.title("FastSLAM Position Error")
plt.grid(True)
plt.savefig("position_error_plot.png")
plt.show()

plt.figure()
plt.plot(steps, n_eff)
plt.xlabel("Step")
plt.ylabel("Effective Sample Size")
plt.title("Particle Filter Effective Sample Size")
plt.grid(True)
plt.savefig("effective_sample_size_plot.png")
plt.show()