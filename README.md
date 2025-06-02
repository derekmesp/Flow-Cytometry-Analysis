Analysis of flow cytometry data from asthmatic donors' various tissue sites.

## preprocessing.py (Updated 4.28.25)
File containing methods for flow_script.ipynb and myeloid_script.ipynb for loading FCS files in python, representing flow cytometry data in a single-cell data structure, and creating visualization of PCA components with donor numbers and group neighborhoods.

## flow_script.ipynb (Updated 4.28.25)
Preprocessing and downstream analysis of flow cytometry data from BLD, BOM, JEL, JLP, LLN, LNG, MLN, and SPL tissues sites of donors. Performs PCA, t-SNE dimensional reductions and creates UMAP visualizations for CD3, CD4, CD8, gdTCR, TCRva filtered groups. 

## myeloid_script.ipynb (Updated 5.28.25)
Preprocessing and downstream analysis of myeloid flow cytometry data from BLD, BOM, JEL, LNG, MLN, and SPL tissues sites of donors. Performs PCA, t-SNE dimensional reductions and creates UMAP visualizations for all cells, CD3-CD19-, CD3-CD19-CD33-, and CD3-CD19-CD33+ filtered groups.

## total_flow.ipynb (Don't remember last update)
Downstream analysis of all combined tisuses for flow_script.ipynb. *Desperately needs to be updated for organization*

## cytokine_assay.ipynb
Very very messy exploratory data analysis of cytokine assay data from asthma donors.
