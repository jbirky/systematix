systematix.decompose: donor decomposition
=========================================

The S3 design matrix and the fit of a donor's SAP flux. See :doc:`/generated`.

.. automodule:: systematix.decompose
   :undoc-members:


Constants
---------

.. autodata:: systematix.decompose.POLY_ORDER

   Highest power of time per segment in the polynomial block.

.. autodata:: systematix.decompose.TAUS

   Ramp timescales (d).

.. autodata:: systematix.decompose.MIN_GAP

   Gaps longer than this (d) start a new ramp.

.. autodata:: systematix.decompose.N_CBV

   Number of CBVs in the CBV block.

.. autodata:: systematix.decompose.T0_JOIN

   Ramp start times within this many days are the same gap in different stars.

.. autodata:: systematix.decompose.BLOCKS

   Design blocks, in the order they are stacked in ``M``.

.. autodata:: systematix.decompose.KIND_ORDER

   Weight-key kinds, in block order.
