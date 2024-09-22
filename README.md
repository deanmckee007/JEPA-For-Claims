# JEPA-For-Claims
Joint Embedding Predictive Architecture for healthcare claims.  A hierarchical approach using within-claims representations and across-claims representation.  Original JEPA paper - https://arxiv.org/abs/2301.08243

Conceptually, the goal here is to generate high quality embeddings for a variety of downstream tasks.  I have a simple prediction head attached that's toggled on/off in config and that should be fine for specializing to a task or expanding to multi-task.  The claims components here are limited to procedures and diagnoses, but anyone implementing this should introduce all of the components relevant for their inference/prediction tasks.

# What can I do with this thing?
Self-supervised models shine where you have a ton of unlabeled data and you want to maximize value from the labeled data you do have.
Claims often have missing/incorrect entries.  Commercial directories and downloads from CMS are often wrong or simply outdated.
Where we have labels we can trust, we can do better.
JEPA for claims allows us to extract representations at a variety of abstractions -
So, want to infer a provider's specialty?  Extract a level 1 provider embedding (within claims) because that captures the procedures and diagnoses providers do.  Optionally also include level 2 provider embeddings since the claims up and downstream from a provider introduce temporal contextual information.  Inferring referring provider is a more obvious use case for level 2 representations.  These features are now the input to the supervised learning model of your choice (or prediction head on this model).

# Notes for use
I have options to toggle all of the level 2 attentional transformations on/off.
For the level 2 prediction block you can choose between GRU, LSTM, and Transformer.  GRU should work well with sequences up to ~50, consider using LSTM beyond that, and transformer if you're feeling spicy.
I'm using my personal laptop for training (yes, yes, I know) - with a GeForce GTX 1660.  It's only got 6GB of memory and I'm able to run with 128 batch size with up to about 125 claims as a context window.

# Future state
I have plans to customize the network further - optionally turning off a level of the hierarchy and also taking a different approach to encoders that will be more in line with what you see with vision transformer set ups (where claims components are channels -> flatten -> linearly project).
