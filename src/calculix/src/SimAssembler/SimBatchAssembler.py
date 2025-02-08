import argparse
import os.path
import shutil
from string import Template
import numpy as np

from SimAssembler import writeParameter, assemble


def writeInput(dx, dy, dz, workingDir, simDir):
    """
    Copy input files and write indenter movement into simulation input file.

    :param dx: movement in x-direction applied to indenter
    :param dy: movement in y-direction applied to indenter
    :param dz: movement in z-direction applied to indenter
    :param workingDir: path to working directory
    :param simDir: path to simulation directory
    :return: None
    """

    os.chdir(workingDir)
    templatesDir = os.path.dirname(os.path.abspath(__file__)) + '/templates'

    # copy input files in simDir
    os.system("cp {} {}".format(templatesDir+"/gelsight_miniSurfaceNodes.nam", simDir+"/gelsight_miniSurfaceNodes.nam"))
    os.system("cp {} {}".format(templatesDir+"/materials.inp", simDir+"/materials.inp"))
    os.system("cp {} {}".format(templatesDir+"/sim.inp", simDir+"/sim.inp"))

    # read input file
    with open(simDir + "/sim.inp", "r") as f:
        all_lines = f.readlines()

    # modify input file
    j = 0
    with open(simDir + "/sim.inp", "w") as f:
        for i, line in enumerate(all_lines, 1):

            if "NindenterFixed,3" in line:
                line = "NindenterFixed,3,," + str(dz) + "\n"

            if "NindenterFixed,1" in line:
                line = "NindenterFixed,1,," + str(dx) + "\n"

            if "NindenterFixed,2" in line:
                line = "NindenterFixed,2,," + str(dy) + "\n"

            f.writelines(line)


def writeSlurm(memCPU, minutes, jobname, idx, workingDir, simDir):
    """
    Write the slurm script to start the simulations on the cluster.

    :param memCPU: memory per cpu in MB
    :param minutes: minutes for every simulation
    :param jobname: name of the job
    :param idx: number of simulations to be run
    :param workingDir: path to working directory
    :param simDir: path to simulation directory
    :return: None
    """

    os.chdir(workingDir)
    templatesDir = os.path.dirname(os.path.abspath(__file__)) + '/templates'

    # copy slurm file to simDir
    os.system("cp {} {}".format(templatesDir+"/calculix.slurm", simDir+"/calculix.slurm"))

    # read slurm file
    with open(simDir + "/calculix.slurm", "r") as f:
        all_lines = f.readlines()

    # modify input file
    with open(simDir + "/calculix.slurm", "w") as f:
        for i, line in enumerate(all_lines, 1):

            if "#SBATCH --mem-per-cpu" in line:
                line = "#SBATCH --mem-per-cpu=" + str(memCPU) + "\n"

            if "#SBATCH -t" in line:
                line = "#SBATCH -t " + '{:02d}:{:02d}:00'.format(*divmod(minutes, 60)) + "\n"

            if "#SBATCH -J" in line:
                line = "#SBATCH -J " + jobname + "_" + str(idx) + "\n"

            if "#SBATCH -a" in line:
                line = "#SBATCH -a 1-" + str(idx) + "\n"

            if "#SBATCH -o" in line:
                line = "#SBATCH -o " + simDir + "%a/%x.out.%A_%a" + "\n"

            if "#SBATCH -e" in line:
                line = "#SBATCH -e " + simDir + "%a/%x.err.%A_%a" + "\n"

            if "cd" in line:
                line = "cd " + simDir + "$SLURM_ARRAY_TASK_ID" + "\n"

            f.writelines(line)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Assemble Simulations.")
    parser.add_argument("--gelsight_miniDir", type=str, default="/obj/gelsight_mini/msh/tetMesh2ndO_field075-3_transinite_FDelaunay_Delaunay/")
    parser.add_argument("--indenterDir", type=str, default="/obj/indenters/sphere/msh/sphere_15/02_tetMesh_2ndOrder_18size/")
    parser.add_argument("--dataDir", type=str, default="/ccx/simulations_sphere_15/sphere_15/")
    parser.add_argument("--simDir", type=str, default="/ccx/simulations_sphere_15/")
    parser.add_argument("--memCPU", type=int, default=7600)
    parser.add_argument("--minutes", type=int, default=120)
    parser.add_argument("--jobname", type=str, default="CCXJobArrayFEATS")
    args = parser.parse_args()

    workingDir = os.getcwd()

    idx = 1

    for file in os.listdir(args.dataDir):

        # if name of file starts with dot, skip
        if file[0] == ".":
            continue

        data = np.load(args.dataDir+file, allow_pickle=True).item()

        simDir_idx = args.simDir + str(idx)

        os.system("mkdir {}".format(simDir_idx))

        os.system("cp {} {}".format(args.dataDir+file, simDir_idx+"/"+file))

        assemble(args.gelsight_miniDir, args.indenterDir, data["x_0"], data["y_0"], data["deg"], simDir_idx)

        writeInput(data["d_x"], data["d_y"], data["d_z"], workingDir, simDir_idx)

        idx += 1

    writeSlurm(args.memCPU, args.minutes, args.jobname, idx-1, workingDir, args.simDir)
