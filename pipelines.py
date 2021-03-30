import cupy

def none(x):
  return None      

class SerialPipelineNode(object):
  def __init__(self, get_x, xforms, get_y=None):
    self.get_x = get_x
    self.xforms = xforms


    if get_y is None:
      get_y = none

    self.get_y = get_y
  
  def generic_fit(self, data, op="fit"):
    xs = self.get_x(data)
    ys = self.get_y(data)
    if ys is not None:
      result = self.xforms[0].fit_transform(xs, y=ys)
      for xf in self.xforms[1:]:
        result = xf.fit_transform(result, y=ys)
    else:
      result = self.xforms[0].fit_transform(xs)
      for xf in self.xforms[1:]:
        result = xf.fit_transform(result)


    if op == "fit":
      return self
    else:
      return result

  def fit(self, data):
    return self.generic_fit(data, "fit")
    
  def fit_transform(self, data):
    return self.generic_fit(data, "fit_transform")
  
  def transform(self, data):
    xs = self.get_x(data)
    result = self.xforms[0].transform(xs)
    for xf in self.xforms[1:]:
      result = xf.transform(result)

    return result


class PrefittedPipelineNode(object):
  def __init__(self, node):
    self.node = node
  
  def fit(self, data, **args):
    pass
  
  def fit_transform(self, data, **args):
    return getattr(self.node, "transform")(data)
  
  def transform(self, data, **args):
    return getattr(self.node, "transform")(data)


class PassThroughPipelineNode(object):
  def __init__(self, get_x):
    self.get_x = get_x
  
  def fit(self, data, **args):
    pass
  
  def fit_transform(self, data, **args):
    return self.get_x(data)
  
  def transform(self, data, **args):
    return self.get_x(data)

class CombiningPipelineNode(object):
  import cupy
  def __init__(self, xfd=None, xkeys=None, combiner=None, ykey=None, dfclass=None, cast=None):
    if xfd is None:
      xfd = dict()
    
    self.cast = cast

    if dfclass is None:
      if 'cudf' in globals():
        dfclass = cudf.DataFrame
      elif 'pd' in globals():
        dfclass = pd.DataFrame
      elif 'pandas' in globals():
        dfclass = pandas.DataFrame
      else:
        assert(False, "I can't figure out which data frame class you want")

    self.xfd = xfd
    self.dfclass = dfclass

    if xkeys is None:
      if ykey is None:
        self.xkeys = list(xfd.keys())
        self.ykey = None
      else:
        self.ykey = ykey
        self.xkeys = list(xfd.keys()) 
        self.xkeys.remove(ykey)
    else:
      self.xkeys = xkeys
      self.ykey = ykey or None
    
    self.output_xkeys = list(xkeys)
    self.combiner=combiner

    if combiner is not None:
      if hasattr(combiner, "predict"):
        self.predict = getattr(self, "unsafe_predict")
      if hasattr(combiner, "predict_proba"):
        self.predict_proba = getattr(self, "unsafe_predict_proba")
  
  def generic_fit(self, data, op="fit", skip_ys=False):
    intermediate_results = self.dfclass()

    for key, spn in self.xfd.items():
      expanded = False

      if skip_ys and key == self.ykey:
        continue
      # nb: this is _always_ fit_transform for efficiency
      tmp = spn.fit_transform(data)
      if self.cast is not None:
        tmp = tmp.astype(self.cast)
      if hasattr(tmp, "values"):
        # XXX: assuming that this is a DF and not an array -- is this safe?
        tmp = tmp.values
      elif hasattr(tmp, "shape"):
        # handle two-dimensional results
        shape = tmp.shape
        if len(shape) == 2:
          _, dims = shape
          stack = cupy.hsplit(tmp, dims)
          for dim in range(dims):
            subkey = '%s__%d' % (key, dim)
            intermediate_results[subkey] = stack[dim]
            if subkey not in self.output_xkeys:
              self.output_xkeys.append(subkey)
          expanded = True
        if key in self.output_xkeys:
          self.output_xkeys.remove(key)
    
      if not expanded:
        intermediate_results[key] = tmp
    
    self.the_df = intermediate_results
    if self.combiner is not None:
      if skip_ys or self.ykey is None:
        return getattr(self.combiner, op)(intermediate_results[self.output_xkeys])
        # return getattr(self.combiner, op)(intermediate_results[self.xkeys].values)
      else:
        return getattr(self.combiner, op)(intermediate_results[self.output_xkeys], y=intermediate_results[self.ykey])
  
  def fit(self, data):
    return self.generic_fit(data, "fit")

  def fit_transform(self, data):
    return self.generic_fit(data, "fit_transform")

  def transform(self, data):
    return self.generic_fit(data, "transform", skip_ys=True)
  
  def unsafe_predict(self, data):
    return self.generic_fit(data, "predict", skip_ys=True)
    
  def unsafe_predict_proba(self, data):
    return self.generic_fit(data, "predict_proba", skip_ys=True)

