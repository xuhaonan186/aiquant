# -*- coding:utf-8 -*-
# Author: quant
# Date: 2023/11/17


class Reweighter:
    def __init__(self, *args, **kwargs):
        """
        To initialize the Reweighter, users should provide specific methods to let reweighter do the reweighting (such as sample-wise, rule-based).
        """
        raise NotImplementedError()

    def reweight(self, data: object) -> object:
        """
        Get weights for data

        Parameters
        ----------
        data : object
            The input data.
            The first dimension is the index of samples

        Returns
        -------
        object:
            the weights info for the data
        """
        raise NotImplementedError(f"This type of input is not supported")

class Dataset:
    def __init__(self, **kwargs):
        """
        init is designed to finish following steps:

        - init the sub instance and the state of the dataset(info to prepare the data)
            - The name of essential state for preparing data should not start with '_' so that it could be serialized on disk when serializing.

        - setup data
            - The data related attributes' names should start with '_' so that it will not be saved on disk when serializing.

        The data could specify the info to calculate the essential data for preparation
        """
        self.config(**kwargs)

    def config(self, **kwargs):
        """
        config is designed to configure and parameters that cannot be learned from the data
        """
        pass

    def prepare(self, **kwargs) -> object:
        """
        The type of dataset depends on the model. (It could be pd.DataFrame, pytorch.DataLoader, etc.)
        The parameters should specify the scope for the prepared data
        The method should:
        - process the data

        - return the processed data

        Returns
        -------
        object:
            return the object or None: cached on disk
        """
        pass


class Scheduler:
    def __init__(self, **kwargs):
        pass

    def config(self, **kwargs):
        """
        config is designed to configure and parameters that cannot be learned from the data
        """
        pass

    def get_scheduler(self, **kwargs):
        pass





