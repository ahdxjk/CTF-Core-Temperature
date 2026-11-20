"""Bound V3.5 internal independent-row chunks without dropping context rows."""
def configure_pfn_chunks(cells=65536):
 from tabpfn import TabPFNRegressor
 if not hasattr(TabPFNRegressor,'_unbounded_initializer'):
  TabPFNRegressor._unbounded_initializer=TabPFNRegressor._initialize_model_variables
  def initialize(self):
   size=TabPFNRegressor._unbounded_initializer(self)
   for model in self.models_:
    if hasattr(model,'inference_chunk_cells'):model.inference_chunk_cells=TabPFNRegressor._runtime_chunk_cells
   return size
  TabPFNRegressor._initialize_model_variables=initialize
 TabPFNRegressor._runtime_chunk_cells=int(cells)
