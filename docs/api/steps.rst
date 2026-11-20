systematix.steps: pointing steps
================================

Kepler pointing steps and the segments they define. See :doc:`/generated`.

.. automodule:: systematix.steps
   :undoc-members:


Constants
---------

.. autodata:: systematix.steps.POINTING_STEP

   ``POS_CORR`` jump (px) that counts as a step.

.. autodata:: systematix.steps.ADJACENT

   Cadences closer than this (d) are compared directly; wider spacings are gaps.

.. autodata:: systematix.steps.STEP_WINDOW

   Across a gap, median positions are taken over this much data (d) on each side.

.. autodata:: systematix.steps.DEFAULT_BITMASK

   lightkurve's ``KeplerQualityFlags.DEFAULT_BITMASK``.
