
class CompositePipeline(object):
    """
    Composite pipelines Pipeline Base Class
    """
    def __init__(self, pipelines):
        """
        TODO: write description
        """
        self.pipelines = pipelines

    def sanitize_parameters(self, **kwargs):
        """
        TODO: write description
        """
        forward_parameters = {}
        for key in kwargs:
            if key == "batch_size":
                forward_parameters["batch_size"] = kwargs["batch_size"]
            if key == "num_workers":
                forward_parameters["num_workers"] = kwargs["num_workers"]
        for pipeline in self.pipelines:
            for dict_params in pipeline.sanitize_parameters(**kwargs):
                forward_parameters.update(dict_params)
        return {}, forward_parameters, {}