package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

config_setting(
    name = "cuda",
    values = {"define": "using_cuda=true"},
)

load(
    "//third_party/tensorflow_lingvo:lingvo.google.bzl",
    "lingvo_pkg_tar",
)

py_library(
    name = "base_runner",
    srcs = ["base_runner.py"],
    deps = [
        ":base_trial",
        ":model_registry",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/core:protos_all_py_pb2",
        "//third_party/tensorflow_lingvo/core:cluster_factory",
        "//third_party/tensorflow_lingvo/core:early_stop",
        "//third_party/tensorflow_lingvo/core:py_utils",
    ],
)

py_library(
    name = "base_trial",
    srcs = ["base_trial.py"],
    deps = [
        "//third_party/tensorflow_lingvo/core:hyperparams",
    ],
)

py_library(
    name = "model_imports_no_params",
    srcs = ["model_imports.py"],
    deps = [
        ":model_registry",
        "//third_party/py/six",
        "//third_party/py/tensorflow",
    ],
)

# Depend on this for access to the model registry with params for all tasks as
# transitive deps.  Only py_binary should depend on this target.
py_library(
    name = "model_imports",
    deps = [
        "//third_party/tensorflow_lingvo:model_imports_no_params",
        "//third_party/tensorflow_lingvo/tasks:all_params",
    ],
)

py_test(
    name = "model_import_test",
    srcs = ["model_import_test.py"],
    deps = [
        ":model_imports_no_params",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "models_test",
    srcs = ["models_test.py"],
    deps = [
        ":model_imports",
        ":model_registry",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow_lingvo/core:base_input_generator",
        "//third_party/tensorflow_lingvo/core:base_model",
        "//third_party/tensorflow_lingvo/core:base_model_params",
    ],
)

py_library(
    name = "model_registry",
    srcs = ["model_registry.py"],
    deps = [
        "//third_party/py/tensorflow",
        "//third_party/tensorflow_lingvo/core:base_model_params",
    ],
)

py_test(
    name = "model_registry_test",
    srcs = ["model_registry_test.py"],
    deps = [
        ":model_registry",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow_lingvo/core:base_input_generator",
        "//third_party/tensorflow_lingvo/core:base_model",
        "//third_party/tensorflow_lingvo/core:base_model_params",
    ],
)

py_library(
    name = "trainer_lib",
    srcs = ["trainer.py"],
    deps = [
        ":base_trial",
        ":model_imports_no_params",
        "//file/placer",
        "//learning/brain/research/babelfish/trainer:base_runner",
        "//third_party/py/IPython:ipython-libs-without-frontend",
        "//third_party/py/numpy",
        "//third_party/py/six",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/core:protos_all_py_pb2",
        "//third_party/tensorflow/core/distributed_runtime/rpc:grpc_runtime",
        "//third_party/tensorflow/python/debug:debug_py",
        "//third_party/tensorflow_lingvo/core:base_model",
        "//third_party/tensorflow_lingvo/core:base_model_params",
        "//third_party/tensorflow_lingvo/core:cluster_factory",
        "//third_party/tensorflow_lingvo/core:metrics",
        "//third_party/tensorflow_lingvo/core:py_utils",
    ],
)

py_binary(
    name = "trainer",
    srcs = [":trainer_lib"],
    deps = [
        ":model_imports",
        ":trainer_lib",
        "//third_party/py/IPython:ipython-libs-without-frontend",
    ],
)

py_test(
    name = "trainer_test",
    srcs = ["trainer_test.py"],
    shard_count = 5,
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "optonly",
    ],
    deps = [
        ":base_trial",
        ":model_registry",
        ":trainer_lib",
        "//third_party/py/six",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow_lingvo/core:base_input_generator",
        "//third_party/tensorflow_lingvo/core:base_layer",
        "//third_party/tensorflow_lingvo/core:base_model",
        "//third_party/tensorflow_lingvo/core:py_utils",
        "//third_party/tensorflow_lingvo/tasks/image:input_generator",
        "//third_party/tensorflow_lingvo/tasks/image/params:mnist",
    ],
)

lingvo_pkg_tar(
    name = "lingvo_trainer_pkg",
    srcs = [":trainer"],
    include_runfiles = True,
    mode = "0644",
    strip_prefix = "/",
)
