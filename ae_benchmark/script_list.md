```
\begin{enumerate}
  \item Metrics analysis of benchmarks(fig 2, fig 3)
  
  Anylysis Execution time, rematerialization count, average recursion depth, fragment
 ratio, and memory footprint of DTR and Megatron-LM in the training of GPT3-7.5B.
    \begin{verbatim}
$ bash bench_analysis_test.sh
    \end{verbatim}
    \vspace{-1em}
    
  \item Quantitation of vertex importance(fig 6).
  
  Test the rematerialization count and average recursion depth of each TR in the Llama2-7B training, by retaining tensors based on the top 5\% of their centrality scores.
    \begin{verbatim}
$ bash dump_centrality_nodes.sh
$ bash bench_centrality.sh
    \end{verbatim}
    \vspace{-1em}

  \item Comparison with non-recomputation systems(table 1).
 
    compares T-Control with state-of-the-art systems without tensor recomputation.
    \begin{verbatim}
$ bash comparison_nonrc.sh
    \end{verbatim}
    \vspace{-1em}
    
  \item Comparison with static TR systems(table 2).
 
    Test the average time and tensor recomputation count(TRC) per iteration of different systems in the  training of various models.
    \begin{verbatim}
$ bash comparison_strc.sh
    \end{verbatim}
    \vspace{-1em}
    
  \item Overall performance(fig 11).
 
    Test performance using the optimal parallelism configuration for each system on 256 GPUs, compared to DeepSpeed ZeRO, Megatron-LM, Zero Bubble, and AdaPipe. This experiment need to be execute on a 64-node cluster, where each node is equipped with 4 NVIDIA A100 GPUs.
    \begin{verbatim}
$ bash submit_dt-control.sh
$ bash megatron.sh
$ cd AdaPipe && bash submit_adapipe.sh
$ cd ../ZeroBubble && submit_zerobubble.sh
$ cd ../DeepSpeed && submit_deepspeed.sh
    \end{verbatim}
    \vspace{-1em}
    
  \item Evaluation with static and dynamic models(table 3).
    
    Test performance on static and dynamic models with various architectures. Compared with the state-of-the art dynamic methods.
    \begin{verbatim}
$ cd ..
$ bash test_other_model.sh
    \end{verbatim}
    \vspace{-1em}
    
  \item Evaluation with different memory budgets(fig 12).
    
    Test the per-step training over head of various dynamic methods under different memory budget ratios.
    \begin{verbatim}
$ bash test_multi_budgets.sh
    \end{verbatim}
    \vspace{-1em}
    
  \item Eviction and rematerialization counts(fig 13).
    
    Test Tensor eviction and rematerialization counts per training step, and the recursion depth of each TR in different methods for Llama2-7B training on 8 A100 GPUs.
    \begin{verbatim}
$ bash test_evre_count.sh
    \end{verbatim}
    \vspace{-1em}
    
  \item Fragmentation ratio and peak memory usage(fig 14).
    
    Test the fragment ratio and peak memory usage of different methods with various memory budgets.
    \begin{verbatim}
$ bash test_fr_pmu.sh
    \end{verbatim}
    \vspace{-1em}
    
  \item System overhead(table 4).
    
    Test The per-iteration system overhead of T-Control in model training, Part of this experiment need to be execute on a 64-node cluster.
    \begin{verbatim}
(On 8*A100 GPU server)
$ bash test_so_static.sh
$ bash test_so_dynamic.sh

(On cluster)
$ bash submit_so_gpt.sh
$ bash submit_so_llama.sh
    \end{verbatim}
    \vspace{-1em}
\end{enumerate}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsection{Evaluation and expected results}

\begin{enumerate}
\item Parse results into CSV files.
\begin{verbatim}
$ python3 ./parse_results.py
\end{verbatim}
\item Generate plots.
\begin{verbatim}
$ python3 ./plot_results.py
\end{verbatim}
\end{enumerate}
```