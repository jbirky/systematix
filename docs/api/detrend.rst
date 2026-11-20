systematix.detrend: detrenders
==============================

A common interface over plug-in filters and linear nuisance models. See :doc:`/detrenders`.

.. automodule:: systematix.detrend
   :undoc-members:


.. autodata:: systematix.detrend.DETRENDERS
   :no-value:

   Detrender classes by name, as used by :func:`make_detrender`: ``quarter_median``,
   ``rolling_median``, ``lk_flatten``, ``poly``, ``ramps``, ``cbv``, ``arc2``. Add
   your own class here to make it available to :func:`make_detrender`.
