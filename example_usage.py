from client import MultiHeadLatentAttentionKvCompressorClient

def main():
    client = MultiHeadLatentAttentionKvCompressorClient()
    res = client.compress_kv_cache_latents(65536, 64, 256)
    print('MLA KV Compressor: ' + res['mla_compression_id'] + ' (Seq Len: ' + str(res['sequence_length_tokens']) + ')')
    print('Memory Savings: ' + str(res['standard_kv_cache_memory_mb']) + ' MB -> ' + str(res['compressed_latent_kv_memory_mb']) + ' MB (' + str(res['kv_compression_ratio_x']) + 'x reduction)')
    print('Reconstruction Fidelity: ' + str(res['attention_reconstruction_fidelity_pct']) + '% | RoPE Decoupled: ' + str(res['rope_decoupled_projection_active']))

if __name__ == '__main__':
    main()
