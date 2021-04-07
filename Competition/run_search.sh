echo("Exploring Effect of Varying K")
python search.py --sparse --output_file outputs/sparse_k0.0_b0.5.candidate.run --k 0.0 --b 0.5
python search.py --sparse --output_file outputs/sparse_k0.1_b0.5.candidate.run --k 0.1 --b 0.5
python search.py --sparse --output_file outputs/sparse_k0.2_b0.5.candidate.run --k 0.2 --b 0.5
python search.py --sparse --output_file outputs/sparse_k0.3_b0.5.candidate.run --k 0.3 --b 0.5
python search.py --sparse --output_file outputs/sparse_k0.4_b0.5.candidate.run --k 0.4 --b 0.5
python search.py --sparse --output_file outputs/sparse_k0.5_b0.5.candidate.run --k 0.5 --b 0.5
python search.py --sparse --output_file outputs/sparse_k0.6_b0.5.candidate.run --k 0.6 --b 0.5
python search.py --sparse --output_file outputs/sparse_k0.7_b0.5.candidate.run --k 0.7 --b 0.5
python search.py --sparse --output_file outputs/sparse_k0.8_b0.5.candidate.run --k 0.8 --b 0.5
python search.py --sparse --output_file outputs/sparse_k0.9_b0.5.candidate.run --k 0.9 --b 0.5
python search.py --sparse --output_file outputs/sparse_k1.0_b0.5.candidate.run --k 1.0 --b 0.5
echo("Done Varying K")
echo("Exploring Effect of Varying B")
python search.py --sparse --output_file outputs/sparse_k0.5_b0.0.candidate.run --k 0.5 --b 0.1
python search.py --sparse --output_file outputs/sparse_k0.5_b0.1.candidate.run --k 0.5 --b 0.1
python search.py --sparse --output_file outputs/sparse_k0.5_b0.2.candidate.run --k 0.5 --b 0.2
python search.py --sparse --output_file outputs/sparse_k0.5_b0.3.candidate.run --k 0.5 --b 0.3
python search.py --sparse --output_file outputs/sparse_k0.5_b0.4.candidate.run --k 0.5 --b 0.4
python search.py --sparse --output_file outputs/sparse_k0.5_b0.5.candidate.run --k 0.5 --b 0.5
python search.py --sparse --output_file outputs/sparse_k0.5_b0.6.candidate.run --k 0.5 --b 0.6
python search.py --sparse --output_file outputs/sparse_k0.5_b0.7.candidate.run --k 0.5 --b 0.7
python search.py --sparse --output_file outputs/sparse_k0.5_b0.8.candidate.run --k 0.5 --b 0.8
python search.py --sparse --output_file outputs/sparse_k0.5_b0.9.candidate.run --k 0.5 --b 0.9
python search.py --sparse --output_file outputs/sparse_k0.5_b1.0.candidate.run --k 0.5 --b 1.0
echo("Done Varying B")
echo("Exporing Effect of original query importance with RM3")
python search.py --sparse --output_file outputs/sparse_original_query0.0.candidate.run --original_query_weight 0.0
python search.py --sparse --output_file outputs/sparse_original_query0.1.candidate.run --original_query_weight 0.1
python search.py --sparse --output_file outputs/sparse_original_query0.2.candidate.run --original_query_weight 0.2
python search.py --sparse --output_file outputs/sparse_original_query0.3.candidate.run --original_query_weight 0.3
python search.py --sparse --output_file outputs/sparse_original_query0.4.candidate.run --original_query_weight 0.4
python search.py --sparse --output_file outputs/sparse_original_query0.5.candidate.run --original_query_weight 0.5
python search.py --sparse --output_file outputs/sparse_original_query0.6.candidate.run --original_query_weight 0.6
python search.py --sparse --output_file outputs/sparse_original_query0.7.candidate.run --original_query_weight 0.7
python search.py --sparse --output_file outputs/sparse_original_query0.8.candidate.run --original_query_weight 0.8
python search.py --sparse --output_file outputs/sparse_original_query0.9.candidate.run --original_query_weight 0.9
python search.py --sparse --output_file outputs/sparse_original_query1.0.candidate.run --original_query_weight 1.0
echo("Done Exploring effect of varying RM3")
