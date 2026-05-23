|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status

.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage

.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Экспериментальная валидация оценок якобиана для итеративных трансформеров
    :Тип научной работы: M1P
    :Автор: Карпеев Глеб Андреевич
    :Научный руководитель: PhD, Грабовой Андрей Валериевич

Abstract
========

Recurrent Transformers are a parameter-efficient approach for modeling deep computations,
but their deployment in iterative settings requires controlled and stable dynamics.
We study the Jacobian dynamics of a full Transformer block with Multi-Head Self-Attention
(MSA), Feed-Forward Networks (FFN), and RMS normalization. We derive an explicit
upper bound on the spectral norm of the full-step Jacobian, showing that RMSNorm acts as a
multiplicative damping factor that prevents gradient explosion. Under mild assumptions, the
Jacobian norm remains *O(1)* in the large-step regime. Our theoretical results, validated on
synthetic and CIFAR-10 data, provide insight into stability properties of recurrent models.

Research publications
===============================
1. Dmitrii Vasilenko, Ilia Stepanov, Vadim Kasiuk, Gleb Karpeev, Andrey Grabovoy.
   *Jacobian Analysis of a Recurrent Transformer Block.* Accepted at AINL 2026.
   Preprint: `paper/Jacobian_Analysis_of_a_Recurrent_Transformer_Block.pdf
   <paper/Jacobian_Analysis_of_a_Recurrent_Transformer_Block.pdf>`_.

Presentations at conferences on the topic of research
================================================
1. AINL 2026 — *Jacobian Analysis of a Recurrent Transformer Block*.

Software modules developed as part of the study
======================================================
1. Core library with model, data, and utilities in `src/ <src/>`_
   (``model.py``, ``data.py``, ``pretrained.py``, ``utils.py``).
2. Experiment scripts in `scripts/ <scripts/>`_:

   - ``exp1_bounds.py`` — validation of Theorem 3.1 bound (fig.\ 1).
   - ``exp2_asymptotic.py`` — asymptotic *O(1)* behavior as ``ν → ∞`` (fig.\ 2).
   - ``exp3_contraction.py`` — contraction dynamics on synthetic & CIFAR-10 (fig.\ 3).
   - ``exp4_tighter_bounds.py`` — hierarchy of tighter Jacobian bounds (fig.\ 4).
   - ``exp5_pretrained.py`` — validation on pretrained DeiT-tiny weights (fig.\ 5).
   - ``exp6_row_aware_bounds.py`` — row-aware outer RMSNorm bound (fig.\ 6).

3. Generated plots in `plots/ <plots/>`_ and CSV results in `results/ <results/>`_.
4. Bound-hierarchy reference note: `results/bound_refinements.md
   <results/bound_refinements.md>`_.

Quickstart
==========

Install dependencies with `uv <https://docs.astral.sh/uv/>`_::

    git clone <repo-url>
    cd Karpeev-MS-Thesis
    uv sync

Run an experiment, e.g.::

    uv run python -m scripts.exp1_bounds
    uv run python -m scripts.exp4_tighter_bounds
