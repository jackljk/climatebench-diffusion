from omegaconf import DictConfig


def get_dims_of_dataset(datamodule_config: DictConfig):
    """Returns the number of features for the given dataset."""
    target = datamodule_config.get("_target_", datamodule_config.get("name"))
    conditional_dim = conditional_non_spatial_dim = 0
    spatial_dims_out = None
    if "oisstv2" in target:
        box_size = datamodule_config.box_size
        input_dim, output_dim, spatial_dims = 1, 1, (box_size, box_size)
    elif "era5" in target:
        if hasattr(datamodule_config, "variables_3d"):
            input_dim = {"3d": len(datamodule_config.variables_3d), "2d": len(datamodule_config.variables_2d)}
            output_dim = {"3d": len(datamodule_config.variables_3d), "2d": len(datamodule_config.variables_2d)}
        else:
            input_dim = len(datamodule_config.input_vars)
            output_dim = len(datamodule_config.output_vars)
        spatial_crop_inputs = datamodule_config.get("spatial_crop_inputs", None)
        if spatial_crop_inputs is not None:
            crop_lat, crop_lon = tuple(spatial_crop_inputs["latitude"]), tuple(spatial_crop_inputs["longitude"])
            if crop_lat == (10, 70) and crop_lon == (190, 310):
                spatial_dims = (80, 40)
            else:
                raise ValueError(f"Unknown spatial crop for inputs: {crop_lat}, {crop_lon}")
        else:
            spatial_dims = (240, 121)  # 1.4 degree resolution

        static_fields = datamodule_config.get("static_fields", []) or []
        conditional_dim = len(static_fields)
        if "lat_lon_embeddings" in static_fields:
            conditional_dim += 2

        if datamodule_config.text_data_path is not None:
            text_emb_type = datamodule_config.text_type
            if text_emb_type in ["tf-idf", None]:
                conditional_non_spatial_dim = 17783  # 7187
            elif text_emb_type == "bert":
                conditional_non_spatial_dim = 768
            elif text_emb_type == "bow":
                conditional_non_spatial_dim = 7187
            elif "llama" in text_emb_type:
                conditional_non_spatial_dim = 4096
            else:
                raise ValueError(f"Unknown text embedding type: {text_emb_type}")

    elif "physical_systems_benchmark" in target:
        if datamodule_config.physical_system == "navier-stokes":
            input_dim, output_dim, spatial_dims = 3, 3, (221, 42)
            conditional_dim = 2
        elif datamodule_config.physical_system == "spring-mesh":
            input_dim, output_dim, spatial_dims = 4, 4, (10, 10)
            conditional_dim = 1
        else:
            raise ValueError(f"Unknown physical system: {datamodule_config.physical_system}")

    elif "kolmogorov" in target:
        # Get the dimensions from the filename
        if "inits" in datamodule_config.filename:
            # e.g. "kolmogorov-N256-n_inits32-T1000.nc"
            _, dim_x, n_trajs, n_timesteps = datamodule_config.filename.strip(".nc").split("-")
            dim_x = dim_y = int(dim_x.strip("N"))
            n_trajs = int(n_trajs.strip("n_inits"))
            n_timesteps = int(n_timesteps.strip("T"))
        else:
            # e.g. "kolmogorov-32-250-256-256.nc"
            n_trajs, n_timesteps, dim_x, dim_y = datamodule_config.filename.strip(".nc").split("-")[1:]

        down_factor = datamodule_config.get("spatial_downsampling_factor", 1)
        spatial_dims = (int(dim_x) // down_factor, int(dim_y) // down_factor)
        n_channels = len(datamodule_config.channels)
        input_dim, output_dim = n_channels, n_channels

    elif "climatebench" in target:
        output_vars = datamodule_config.get("output_vars")
        additional_vars = datamodule_config.get("additional_vars", [])
        input_dim = 4
        if additional_vars is not None:
            input_dim += len(additional_vars)
        output_dim = 1 if isinstance(output_vars, str) else len(output_vars)
        # conditional_dim = len(additional_vars)
        spatial_dims = (96, 144)  # input spatial dimensions
        if "climatebench_daily" in target:
            spatial_dims_out = (192, 288)  # output spatial dimensions
        else:
            spatial_dims_out = spatial_dims

    elif "fv3gfs" in target:
        input_dim = len(datamodule_config.in_names)
        output_dim = len(datamodule_config.out_names)
        spatial_dims = (180, 360)
        conditional_dim = len(datamodule_config.forcing_names) if datamodule_config.forcing_names is not None else 0

    elif "debug_datamodule" in target:
        input_dim = output_dim = datamodule_config.channels
        spatial_dims = (datamodule_config.height, datamodule_config.width)

    else:
        raise ValueError(f"Unknown dataset: {target}")
    return {
        "input": input_dim,
        "output": output_dim,
        "spatial_in": spatial_dims,
        "spatial_out": spatial_dims_out if spatial_dims_out is not None else spatial_dims,
        "conditional": conditional_dim,
        "conditional_non_spatial_dim": conditional_non_spatial_dim,
    }
