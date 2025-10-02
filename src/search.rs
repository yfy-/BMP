use crate::index::forward_index::block_score;
use crate::index::forward_index::BlockForwardIndex;
use crate::index::posting_list::PostingListIterator;
use crate::query::cursor::DocId;
use crate::query::cursor::{RangeMaxScore, RangeMaxScoreCursor};
use crate::query::live_block;
use crate::query::topk_heap::TopKHeap;
use crate::util::progress_bar;
use std::arch::x86_64::_mm_prefetch;
use std::time::Instant;

pub fn b_search(
    queries: Vec<Vec<PostingListIterator>>,
    forward_index: &BlockForwardIndex,
    k: usize,
    alpha: f32,
    terms_r: f32,
    num_runs: usize,
) -> Vec<TopKHeap<u16>> {
    b_search_verbose(queries, forward_index, k, alpha, terms_r, num_runs, true)
}

pub fn b_search_verbose(
    mut queries: Vec<Vec<PostingListIterator>>,
    forward_index: &BlockForwardIndex,
    k: usize,
    alpha: f32,
    terms_r: f32,
    num_runs: usize,
    verbose: bool,
) -> Vec<TopKHeap<u16>> {
    let mut results: Vec<TopKHeap<u16>> = Vec::new();

    let progress = if verbose {
        Some(progress_bar(
            "Forward index-based search",
            queries.len() * num_runs,
        ))
    } else {
        None
    };

    let mut search_elapsed: Vec<u128> = vec![];
    let mut buckets: Vec<Vec<u32>> =
        (0..=2usize.pow(16)).map(|_| Vec::new()).collect();

    for run in 0..=num_runs {
        for query in queries.iter_mut() {
            let start_search: Instant = Instant::now();
            let total_terms = query.len();
            let terms_to_keep = (total_terms as f32 * terms_r).ceil() as usize;
            query.sort_by(|a, b| {
                b.term_weight().partial_cmp(&a.term_weight()).unwrap()
            });
            // Keep only the top N terms
            query.truncate(terms_to_keep);

            let query_weights: Vec<_> =
                query.iter().map(|post| post.term_weight()).collect();

            let query_ranges: Vec<_> =
                query.iter().map(|post| post.range_max_scores()).collect();
            let mut query_ranges_raw = Vec::new();
            let mut query_ranges_compressed = Vec::new();
            for qr in query_ranges {
                match qr {
                    RangeMaxScore::Compressed(compressed) => {
                        query_ranges_compressed.push(compressed)
                    }
                    RangeMaxScore::Raw(raw) => query_ranges_raw.push(raw),
                };
            }

            let mut query_vec = query
                .iter()
                .map(|&pl| (pl.term_id() as u16, pl.term_weight() as u8))
                .collect::<Vec<_>>();
            query_vec.sort_by_key(|e| e.0);
            let threshold = query
                .iter()
                .map(|&pl| pl.kth(k) as u16 * pl.term_weight() as u16)
                .max()
                .unwrap_or(0);

            let run_compressed = query_ranges_compressed.len() > 0;
            let upper_bounds = match run_compressed {
                true => live_block::compute_upper_bounds(
                    &query_ranges_compressed,
                    &query_weights,
                    forward_index.data.len(),
                ),
                false => live_block::compute_upper_bounds_raw(
                    &query_ranges_raw,
                    &query_weights,
                    forward_index.data.len(),
                ),
            };

            let mut topk = TopKHeap::with_threshold(k, threshold as u16);
            buckets.iter_mut().for_each(std::vec::Vec::clear);
            upper_bounds.iter().enumerate().for_each(|(range_id, &ub)| {
                if ub > threshold {
                    buckets[ub as usize].push(range_id as u32);
                }
            });

            let mut ub_iter = buckets.iter_mut().enumerate().rev().flat_map(
                |(outer_idx, inner_vec)| {
                    inner_vec.iter_mut().map(move |val| (outer_idx, val))
                },
            );

            let (mut current_ub, mut current_block) = ub_iter.next().unwrap();
            unsafe {
                _mm_prefetch(
                    forward_index.data.as_ptr().add(*current_block as usize)
                        as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
            for (next_ub, next_block) in ub_iter {
                unsafe {
                    _mm_prefetch(
                        forward_index.data.as_ptr().add(*next_block as usize)
                            as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                }

                let offset = *current_block as usize * forward_index.block_size;

                let res = block_score(
                    &query_vec,
                    &forward_index.data[*current_block as usize],
                    forward_index.block_size,
                );

                for (doc_id, &score) in res.iter().enumerate() {
                    topk.insert(DocId(doc_id as u32 + offset as u32), score);
                }

                if topk.threshold() as f32 > current_ub as f32 * alpha {
                    break;
                }
                current_block = next_block;
                current_ub = next_ub;
            }
            if run == 0 {
                results.push(topk.clone());
            } else {
                search_elapsed.push(start_search.elapsed().as_micros());
            }

            if let Some(progress_bar) = &progress {
                progress_bar.inc(1);
            }
        }
    }
    if let Some(progress_bar) = &progress {
        progress_bar.finish();
    }

    search_elapsed.sort();
    let mut total_elapsed: f64 = 0f64;
    for se in search_elapsed {
        total_elapsed += se as f64;
    }

    let num_elapsed = search_elapsed.len();
    let avg_elapsed = total_elapsed / num_elapsed as f64;
    let median_elapsed = search_elapsed[num_elapsed / 2];
    let p95_elapsed = search_elapsed[num_elapsed * 95 / 100];
    let p99_elapsed = search_elapsed[num_elapsed * 99 / 100];

    if verbose {
        eprintln!("avg_elapsed={:.2}", avg_elapsed);
        eprintln!("p50_elapsed={:.2}", median_elapsed);
        eprintln!("p95_elapsed={:.2}", p95_elapsed);
        eprintln!("p99_elapsed={:.2}", p99_elapsed);
    }

    results
}
