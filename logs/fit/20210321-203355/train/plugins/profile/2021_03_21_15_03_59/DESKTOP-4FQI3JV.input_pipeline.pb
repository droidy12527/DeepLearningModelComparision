	??&S?s@??&S?s@!??&S?s@	??-ZS????-ZS??!??-ZS??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??&S?s@??_vO??ApΈ???s@Y??/?$??*	     d?@2F
Iterator::Modelp_?Q??!??v*"?V@)e?X???1?v
o!V@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??j+????!??a??
@)?
F%u??18*K??^@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateK?=?U??!??%V@)??_?L??13?F????:Preprocessing2U
Iterator::Model::ParallelMapV2U???N@??!??HȬ??)U???N@??1??HȬ??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicen??t?!^?a?'???)n??t?1^?a?'???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?
F%u??!8*K??^#@)U???N@s?1??HȬ??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensory?&1?l?!h?? ?Z??)y?&1?l?1h?? ?Z??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap"??u????!/?<
@)ŏ1w-!_?1?؟/??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??-ZS??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??_vO????_vO??!??_vO??      ??!       "      ??!       *      ??!       2	pΈ???s@pΈ???s@!pΈ???s@:      ??!       B      ??!       J	??/?$????/?$??!??/?$??R      ??!       Z	??/?$????/?$??!??/?$??JCPU_ONLYY??-ZS??b 