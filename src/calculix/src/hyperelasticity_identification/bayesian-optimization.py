import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

matplotlib.rcParams.update({"font.size": 14})


def read_dat(dat_filename):
    """
    Read the last value of the total force in z-direction from a .dat file.

    :param dat_filename: result file from CalculiX (.dat)
    :return: fz: total force in z-direction
    """

    # check that file exists
    assert os.path.isfile(dat_filename), "File does not exist: {}".format(dat_filename)

    # only save values from last time step
    save = False

    with open(dat_filename, "r") as f:
        for line in f:
            # skip lines until time step is reached
            if "total force (fx,fy,fz) for set NGELSIGHT_MINIVOLUME and time  0.1000000E+01" in line:
                save = not save
                next(f)
                continue

            # make sure that values are saved only from last time step
            if save:
                # make sure that line is not empty
                if line.strip():
                    fz = float(line.split()[-1])
                    break

    return fz


def write_inp(folder_name, x1, x2, dz):
    """
    Make a new copy of the templates folder with modified material values.

    :param folder_name: path to the templates folder
    :param x1: material parameter 1
    :param x2: material parameter 2
    :param dz: displacement in z-direction
    :return: folder_name_copy: path to the new folder
    """

    # round values
    x1 = round(x1, 7)
    x2 = round(x2, 1)

    # check that folder exists
    assert os.path.isdir(folder_name), "Folder does not exist: {}".format(folder_name)

    # create copy of folder
    if nonlinear == False:
        folder_name_copy = folder_name.split("templates/")[0] + "e_" + str(dz).replace(".", "") + "_" + str(x1).replace(".", "") + "_" + str(x2).replace(".", "") + "/"
    elif nonlinear == True:
        folder_name_copy = folder_name.split("templates/")[0] + "he_" + str(dz).replace(".", "") + "_" + str(x1).replace(".", "") + "_" + str(x2).replace(".", "") + "/"
    os.system("mkdir -p {}".format(folder_name_copy))
    os.system("cp {} {}".format(folder_name + str("sim.inp"), folder_name_copy))
    os.system("cp {} {}".format(folder_name + str("materials.inp"), folder_name_copy))

    # read materials file
    with open(folder_name_copy + "materials.inp", "r") as f:
        all_lines = f.readlines()

    # modify materials file
    j = 0
    with open(folder_name_copy + "materials.inp", "w") as f:
        for i, line in enumerate(all_lines, 1):

            if j == 2:
                line = str(x1) + "," + str(x2) + "\n"
                j = 0
            elif j == 1:
                j = 2
            elif "*MATERIAL, NAME=Silicone" in line:
                j = 1

            f.writelines(line)

    # read input file
    with open(folder_name_copy + "sim.inp", "r") as f:
        all_lines = f.readlines()

    # modify input file
    j = 0
    with open(folder_name_copy + "sim.inp", "w") as f:
        for i, line in enumerate(all_lines, 1):

            if j == 1:
                line = "*INCLUDE, INPUT=" + folder_name_copy + "materials.inp" + "\n"
                j = 0
            elif "** Materials" in line:
                j = 1
            elif j == 2:
                if "*BOUNDARY" in line:
                    j = 3
            elif j == 3:
                line = "NindenterFixed,3,,-" + str(dz) + "\n"
                j = 0
            elif "** Normal Simulation Step" in line:
                j = 2

            f.writelines(line)

    return folder_name_copy


def objective_function(x1):
    """
    Objective function for the Bayesian optimization.

    :param x1: material parameter 1
    :return: loss: negative squared difference between predicted and ground truth force
    """

    x2=0.0
    templates_folder = "templates/"

    dz_1 = 0.5
    fz_gt_1 = -1.6503

    dz_2 = 1.0
    fz_gt_2 = -5.9641

    dz_3 = 1.5
    fz_gt_3 = -13.5257

    dz_4 = 2.0
    fz_gt_4 = -23.2532

    # create new inp file with different material values
    folder_name_copy_1 = write_inp(templates_folder, x1, x2, dz_1)
    folder_name_copy_2 = write_inp(templates_folder, x1, x2, dz_2)
    folder_name_copy_3 = write_inp(templates_folder, x1, x2, dz_3)
    folder_name_copy_4 = write_inp(templates_folder, x1, x2, dz_4)

    # run FEM with new inp file
    os.system("ccx {}sim".format(folder_name_copy_1))
    os.system("ccx {}sim".format(folder_name_copy_2))
    os.system("ccx {}sim".format(folder_name_copy_3))
    os.system("ccx {}sim".format(folder_name_copy_4))

    try:
        fz_pred_1 = read_dat(folder_name_copy_1 + "sim.dat")
    except:
        fz_pred_1 = 0

    try:
        fz_pred_2 = read_dat(folder_name_copy_2 + "sim.dat")
    except:
        fz_pred_2 = 0

    try:
        fz_pred_3 = read_dat(folder_name_copy_3 + "sim.dat")
    except:
        fz_pred_3 = 0

    try:
        fz_pred_4 = read_dat(folder_name_copy_4 + "sim.dat")
    except:
        fz_pred_4 = 0

    loss_1 = - (np.abs(fz_gt_1) - np.abs(fz_pred_1))**2
    loss_2 = - (np.abs(fz_gt_2) - np.abs(fz_pred_2))**2
    loss_3 = - (np.abs(fz_gt_3) - np.abs(fz_pred_3))**2
    loss_4 = - (np.abs(fz_gt_4) - np.abs(fz_pred_4))**2

    loss = 1/4 * (loss_1 + loss_2 + loss_3 + loss_4)

    return loss


def plot_optimizer(optimizer):
    """
    Create a plot of the target function and the optimizer's surrogate model.

    :param optimizer: Bayesian optimization object
    :return: None
    """

    # create x values
    x = np.linspace(0.05, 0.09, 1000)

    # predict mean and standard deviation
    mean, sigma = optimizer._gp.predict(x.reshape(-1, 1), return_std=True)

    # save results
    results = {
        "x": x,
        "mean": mean,
        "sigma": sigma,
        "params": optimizer.space.params,
        "target": optimizer.space.target
    }

    np.save("results.npy", results)

    # create figure
    plt.figure(figsize=(8, 5))
    plt.grid()

    # plot target function
    plt.plot(x, mean)
    plt.fill_between(x, mean + sigma, mean - sigma, alpha=0.1)
    plt.scatter(optimizer.space.params.flatten(), optimizer.space.target, c="red", s=50, zorder=10)

    plt.xlabel("c10")
    plt.ylabel("O(c10)")

    plt.savefig("bo_target_function.pdf", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":

    # define global variable
    global nonlinear
    nonlinear = True

    # Bounded region of parameter space
    if nonlinear == False:
        pbounds = {"x1": (0.01, 1), "x2": (0.01, 0.5)}
    elif nonlinear == True:
        pbounds = {"x1": (0.05, 0.09)} #, "x2": (0.0, 0.0)}

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )

    logger = JSONLogger(path="./logs.log")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=2,
        n_iter=25,
    )

    print(optimizer.max)

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    plot_optimizer(optimizer)
