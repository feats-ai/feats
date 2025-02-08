import argparse
import os.path
import shutil
from string import Template
from tkinter.filedialog import askdirectory


def writeParameter(gelsight_miniDir, indenterDir, offsetX, offsetY, rotZ, templateDir, dest):
    """
    Writes the positional parameters of the indenenter and gelsight_mini mesh to the parameters file.

    :param gelsight_miniDir: path to gelsight_mini mesh
    :param indenterDir: path to indenter mesh
    :param offsetX: offset in x-direction applied to indenter
    :param offsetY: offset in y-direction applied to indenter
    :param rotZ: optional rotation around the z axis of the indenter (in degrees)
    :param templateDir: path to template directory
    :param dest: destination of the parameter file
    :return: None
    """

    d = {
        'indenterDir': indenterDir,
        'gelsight_miniDir': gelsight_miniDir,
        'offsetX': offsetX,
        'offsetY': offsetY,
        'rotZ': rotZ
    }

    writef = open(dest, 'w')

    with open(templateDir+'/parameters.template', 'r') as f:
        src = Template(f.read())
        result = src.substitute(d)
        writef.write(result)

    writef.close()


def assemble(gelsight_miniDir, indenterDir, offsetX, offsetY, rotZ, simDir):
    """
    Assembles the gelsight_mini and indenter mesh with a given indenter position.

    :param gelsight_miniDir: path to gelsight_mini mesh
    :param indenterDir: path to indenter mesh
    :param offsetX: offset in x-direction applied to indenter
    :param offsetY: offset in y-direction applied to indenter
    :param rotZ: optional rotation of the indenter around the z axis (in degrees)
    :param simDir: destination of the simulation
    :return: None
    """

    paramFile = simDir + '/parameters.fbd'
    asmblFile = simDir + '/assembleContactTrial.fbd'

    if not os.path.exists(simDir):
        os.mkdir(simDir)
    else:
        if os.path.isfile(paramFile):
            os.system('rm {}'.format(paramFile))
        if os.path.isfile(asmblFile):
            os.system('rm {}'.format(asmblFile))

    templatesDir = os.path.dirname(os.path.abspath(__file__)) + '/templates'
    absGelsight_miniDir = os.path.abspath(gelsight_miniDir)
    absIndenterDir = os.path.abspath(indenterDir)

    writeParameter(absGelsight_miniDir, absIndenterDir, offsetX, offsetY, rotZ, templatesDir, paramFile)
    os.system('cp {} {}'.format(templatesDir+'/assembleContactTrial.template', asmblFile))

    # run assemble script and remove temp files
    os.chdir(simDir)
    os.system('cgx -bg assembleContactTrial.fbd')
    os.system('rm {} {}'.format(paramFile, asmblFile))


if __name__ == "__main__":
   parser = argparse.ArgumentParser(prog = 'Contact Trial Mesh Assembler',
                                    description = 'Combines a GelSight and indenter Mesh with a given indenter Position',
                                    epilog = '© 2024 Erik Helmut & Luca Dziarski')
   #parse meshes
   parser.add_argument("-g", "--gelDir", dest="gelsight_miniDir", required=False,
                    help="Directory of the gelsight_mini mesh", metavar="DIR")
   parser.add_argument("-i", "--indenterDir", dest="indenterDir", required=False,
                    help="Directory of the indenter mesh", metavar="DIR")

   #parse sim parameter
   parser.add_argument("-s", "--simDir", dest="simDir", required=False,
                    help="Directory of the simulation", metavar="DIR")
   parser.add_argument("-x", "--offsetX", dest="offsetX", required=False, default=0.0, type=float,
                    help="Offset in x-direction applied to indenter", metavar="DOUBLE")
   parser.add_argument("-y", "--offsetY", dest="offsetY", required=False, default=0.0, type=float,
                    help="Offset in y-direction applied to indenter", metavar="DOUBLE")
   parser.add_argument("-r", "--rotation", dest="rotZ", required=False, default=0.0, type=float,
                    help="Rotation of indenter around z axis (in degrees)", metavar="DEG")

   args = parser.parse_args()

   # use gui dialog for missing arguments
   if args.gelsight_miniDir is None:
      gelsight_miniDir = askdirectory(title='Choose gelsight_mini mesh directory.')
   else:
      gelsight_miniDir = args.gelsight_miniDir
   if args.indenterDir is None:
      indenterDir = askdirectory(title='Choose indenter mesh directory.')
   else:
      indenterDir = args.indenterDir
   if args.simDir is None:
      simDir = askdirectory(title='Chose destination of simulation.')
   else:
      simDir = args.simDir

   # perform type checking
   if not os.path.isdir(gelsight_miniDir):
      raise TypeError('Given gel directory: \"{}\" is not a valid directory'.format(args.gelsight_miniDir))
   if not os.path.isdir(indenterDir):
      raise TypeError('Given indenter directory: \"{}\" is not a valid directory'.format(args.indenterDir))

   # call assembly
   assemble(gelsight_miniDir, indenterDir, args.offsetX, args.offsetY, args.rotZ, simDir)
