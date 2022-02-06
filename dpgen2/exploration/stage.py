from dpgen2.exploration.task import ExplorationTaskGroup
from abc import (
    ABC,
    abstractmethod,
)
from dpgen2.constants import (
    lmp_conf_name, 
    lmp_input_name,
    model_name_pattern,
)
from dpgen2.exploration.task import (
    ExplorationTaskGroup,
    ExplorationTask,
)
from typing import (
    List,
)
from .group import ExplorationGroup

class ExplorationStage():
    """
    The exploration stage.

    """

    def __init__(self):
        self.clear()

    def clear(self):
        """
        Clear all exploration group.

        """
        self.explor_groups = []

    def add_group(
            self,
            grp : ExplorationGroup,
    ):
        """
        Add an exploration group
        
        Parameters
        ----------
        grp: ExplorationGroup
            The added exploration group

        """
        self.explor_groups.append(grp)
        return self

    def make_task(
            self,
    )->ExplorationTaskGroup:
        """
        Make the LAMMPS task group.        

        Returns
        -------
        task_grp: ExplorationTaskGroup
            The returned lammps task group. The number of tasks is equal to
            the summation of task groups defined by all the exploration groups
            added to the stage.

        """

        lmp_task_grp = ExplorationTaskGroup()
        for ii in self.explor_groups:
            # lmp_task_grp.add_group(ii.make_task())
            lmp_task_grp += ii.make_task()
        return lmp_task_grp


