.. highlight:: rst

Introduction
=============================

**iScore: a MPI supported software for ranking protein-protein docking models based on a random walk graph kernel and
support vector machines**

The software supports the publication of the following articles:

C. Geng *et al.*, *iScore: A novel graph kernel-based function for scoring protein-protein docking models*, bioRxiv 2018,  https://doi.org/10.1101/498584


iScore uses a support vector machine (SVM) approach to rank protein-protein docking models using their interface information. Each interface is represented as a bipartite graph, in which each node represents a contact residue and each edge denotes the two nodes are close to each other in 3D space (the current cutoff is 6 A). Currently, edges are not labelled, and each node is labeled with 20 by 1 vector from the Position Specific Scoring Matrix (PSSM) of the corresponding residue.

To measure the similarity between two graphs, iScore use a random walk graph kernel (RWGK) approach. The graph kernel matrix for all graph pairs is then used as input of the SVM model to either train the model on a training set or use a pretrained model to rank new protein-protein docking models.

.. image :: comp.png







