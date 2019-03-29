import loss_landscapes.backends.torch.backend
import loss_landscapes.backends.torch.ops

TORCH_BACKEND = loss_landscapes.backends.torch.backend.TorchBackend()


def compute_line(model, direction, distance, steps, library, evaluation_f, agent=None, model_set_f=None):
    # dispatches computation along a line to the appropriate backend
    if library == 'torch':
        return TORCH_BACKEND.line(model, direction, distance, steps, evaluation_f, agent, model_set_f)
    else:
        raise ValueError('Invalid library flag ' + str(library) + ' passed to _compute_line')


def compute_plane(model, direction_one, direction_two, distance_one, distance_two,
                  steps, library, evaluation_f, center=False, agent=None, model_set_f=None):
    # dispatches computation along a plane to the appropriate backend
    if library == 'torch':
        return TORCH_BACKEND.plane(model, direction_one, direction_two, distance_one, distance_two, steps,
                                   evaluation_f, center, agent, model_set_f)
    else:
        raise ValueError('Invalid library flag ' + str(library) + ' passed to _compute_plane')


def get_parameters(model, library):
    # dispatches model parameter extraction to the appropriate backend
    if library == 'torch':
        return list(model.parameters())
    else:
        raise ValueError('Invalid library flag ' + str(library) + ' passed to _get_parameters')


def sample_uniform_like(vector, library, unit_vector=True):
    # dispatches random sampling of a vector from uniform distribution to the appropriate backend
    if library == 'torch':
        return TORCH_BACKEND.sample_uniform_like(vector, unit_vector)
    else:
        raise ValueError('Invalid library flag ' + str(library) + ' passed to _sample_uniform_like')


def get_advanced_normalized_vector(direction, model_parameters, norm_type, library):
    # dispatches application of filter normalization (and other norms) to the appropriate backend
    if library == 'torch':
        return TORCH_BACKEND.get_normalized_vector(direction, model_parameters, norm_type)
    else:
        raise ValueError('Invalid library flag ' + str(library) + ' passed to _get_normalized_vector')


def get_unit_vector(vector, library):
    # dispatches computing the unit vector of a direction to the appropriate backend
    if library == 'torch':
        return loss_landscapes.backends.torch.ops.unit_vector(vector)
    else:
        raise ValueError('Invalid library flag ' + str(library) + ' passed to _get_unit_vector')


def get_displacement(point_a, point_b, library):
    # dispatches computing the displacement between two points to the appropriate backend
    if library == 'torch':
        return loss_landscapes.backends.torch.ops.vector_subtraction(point_b, point_a)
    else:
        raise ValueError('Invalid library flag ' + str(library) + ' passed to _get_displacement')


def get_l2_norm(vector, library):
    # dispatches computing the length of a vector to the appropriate backend
    if library == 'torch':
        return loss_landscapes.backends.torch.ops.l2_norm(vector)
    else:
        raise ValueError('Invalid library flag ' + str(library) + ' passed to _get_l2_norm')
