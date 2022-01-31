import numpy as np
import mdtraj as md
import plotly.express as px
import pandas as pd

traj = md.load('clear_traj.pdb')
atoms, bonds = traj.topology.to_dataframe()

def rama_plot(phi_index, psi_index):
    """ Plot the Ramachandran scatter defined by the input list
        phi_index : list of integers defining phi angle
        psi_index : list of integers defining psi angle """

    angles = md.compute_dihedrals(traj, [phi_index, psi_index])
    df=pd.DataFrame(angles, columns=['phi', 'psi'] )
    df['Time'] = range(2500)
    fig = px.scatter(df, x='phi', y='psi', color= 'Time', title= 'Dihedral Map: Alanine dipeptide')
    return fig

def rama_frame(phi_index, psi_index):
    """ Plot phi according to psi at a time that the user can define by using the slider

        phi_index : list of integers defining phi angle
        psi_index : list of integers defining psi angle """

    angles = md.compute_dihedrals(traj, [phi_index, psi_index])
    df=pd.DataFrame(angles, columns=['phi', 'psi'] )
    df['Time'] = range(2500)
    fig = px.scatter(df, x='phi', y='psi', color= 'Time', animation_frame='Time')

    return fig