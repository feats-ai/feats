# CalculiX Files
## [obj/](https://github.com/feats-ai/feats/edit/main/src/calculix/obj)
Models and meshes for the gel/indenter. The files are organized using the following structure: 
```
📂 {Object}/                    # Parentfolder for specific object (e.g. Sphere)
    📂 model/                   # Contains all 3D-Models of the object (e.g. sphere_15 -> 15mm diameter)
        📄 {model1}
        📄 {model2}
        📄 ...
    📂 msh/                     # Contains all meshes for corresponding models 
        📂 {model1}/            # e.g. sphere_15
            📄 setup.fbd        # calculix batch file to setup mesh in simulation
            📂 {mesh1}          3 e.g. 01_tetMesh_2ndOrder_18size
                📂 ccx          # calculix .inp files of mesh
                    ...
                📄 {mesh1}.geo  # gmsh .geo file containing mesh settings
            📂 {mesh2}
            📂 ...
        📂 {model2}/
            📄 setup.fbd
            ...
        📂 ...
```
**Remark:** gelsight mesh folder does not contain subfolders for different models as there is only one.
