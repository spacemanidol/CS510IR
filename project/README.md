\documentclass[sigplan,screen]{acmart}
%% NOTE that a single column version is required for 
%% submission and peer review. This can be done by changing
%% the \doucmentclass[...]{acmart} in this template to 
%% \documentclass[manuscript,screen,review]{acmart}
%% 
%% To ensure 100% compatibility, please check the white list of
%% approved LaTeX packages to be used with the Master Article Template at
%% https://www.acm.org/publications/taps/whitelist-of-latex-packages 
%% before creating your document. The white list page provides 
%% information on how to submit additional LaTeX packages for 
%% review and adoption.
%% Fonts used in the template cannot be substituted; margin 
%% adjustments are not allowed.
%%
%% \BibTeX command to typeset BibTeX logo in the docs
\AtBeginDocument{%
  \providecommand\BibTeX{{%
    \normalfont B\kern-0.5em{\scshape i\kern-0.25em b}\kern-0.8em\TeX}}}

\begin{document}
\title{CS510 Project Proposal: Exploring Effect of Model pruning for Information Retrieval}


\author{Daniel Campos}
\email{dcampos3@illinois.edu}


%%
%% The abstract is a short summary of the work to be presented in the
%% article.
\begin{abstract}
In the last few decades search engines like Google, Bing, Baidu, and Yandex have become the primary way that people around the world interact with information. Their constant and diverse usage has made these engines ideal sources for training data like document co-clicks. Using the ORCAS dataset we seek to explore the intersection of Data Mining and Deep Learning and answer the question: Can document co-clicks be used to learn similarity between concepts?
\end{abstract}

\maketitle
\input{1description}
\input{2ideas.tex}
\input{3plan}
\input{4papers}
\bibliographystyle{ACM-Reference-Format}
\bibliography{bibliography}
%\appendix
\end{document}
\endinput

