import h5py
import pickle

def write_plan_to_hdf5(file_name, plan):
    if file_name[-5:] != '.hdf5':
        file_name += '.hdf5'

    hdf5_file = h5py.File(file_name, 'w')
    plan_group = hdf5_file.create_group('plan')
    action_group = plan_group.create_group('actions')
    param_group = plan_group.create_group('params')

    for param in plan.params.values():
        add_param_to_group(param_group, param)
    
    for action in plan.actions:
        add_action_to_group(action_group, action)


def add_action_to_group(group, action):
    action_group = group.create_group(action.name)
    action_group['name'] = action.name
    action_group['active_ts'] = action.active_timesteps
    action_group['params'] = map(lambda p: p.name, action.params)

    for i in range(len(action.preds)):
        pred = action.preds[i]
        add_action_pred_to_group(action_group, pred, i)


def add_action_pred_to_group(group, pred, index):
    pred_group = group.create_group(str(index))
    pred_group['negated'] = pred['negated']
    pred_group['hl_info'] = pred['hl_info']
    pred_group['active_ts'] = pred['active_timesteps']
    add_pred_to_group(pred_group, pred['pred'])


def add_pred_to_group(group, pred):
    print '***** - 2'
    pred_group = group.create_group('pred')
    pred_group['class_path'] = str(type(pred)).split("'")[1]
    pred_group['name'] = pred.name
    print '***** - 3'
    # Note: Assumes parameter names and types will be at most 64 characters long
    param_dset = pred_group.create_dataset('params', (len(pred.params),), dtype='S64')
    param_types_dset = pred_group.create_dataset('param_types', (len(pred.params),), dtype='S64')
    for i in range(len(pred.params)):
        param_dset[i] = pred.params[i].name
        param_types_dset[i] = pred.params[i].get_type()

    print '***** - 4'


def add_param_to_group(group, param):
    param_group = group.create_group(param.name)
    param_group['name'] = param.name

    geom = None
    if hasattr(param, 'geom') and param.geom:
        geom = param.geom
        param.geom = None
        add_geom_to_group(param_group, geom)

    or_body = None
    if hasattr(param, 'openrave_body'):
        or_body = param.openrave_body
        del param.openrave_body

    try:
        param_group['data'] = pickle.dumps(param)
    except pickle.PicklingError:
        print "Could not pickle {0}.".format(param.name)

    if geom:
        param.geom = geom

    if or_body:
        param.openrave_body = or_body


def add_geom_to_group(group, geom):
    geom_group = group.create_group('geom')
    geom_group['class_path'] = str(type(geom)).split("'")[1]
    try: 
        geom_group['data'] = pickle.dumps(geom)
    except pickle.PicklingError:
        print "Could not pickle {0}.".format(type(geom))
        geom_group['data'] = pickle.dumps(None)


def read_from_hdf5(file_name):
    pass
