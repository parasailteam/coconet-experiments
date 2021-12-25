#include <ATen/Parallel.h>
#include <gtest/gtest.h>
#include <cstring>

#include <c10/util/irange.h>
#include <libgen.h>
#include <torch/csrc/deploy/deploy.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <future>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  return rc;
}

void compare_torchpy_jit(const char* model_filename, const char* jit_filename) {
  // Test
  torch::deploy::InterpreterManager m(1);
  torch::deploy::Package p = m.load_package(model_filename);
  auto model = p.load_pickle("model", "model.pkl");
  at::IValue eg;
  {
    auto I = p.acquire_session();
    eg = I.self.attr("load_pickle")({"model", "example.pkl"}).toIValue();
  }

  at::Tensor output = model(eg.toTuple()->elements()).toTensor();

  // Reference
  auto ref_model = torch::jit::load(jit_filename);
  at::Tensor ref_output =
      ref_model.forward(eg.toTuple()->elements()).toTensor();

  ASSERT_TRUE(ref_output.allclose(output, 1e-03, 1e-05));
}

const char* simple = "torch/csrc/deploy/example/generated/simple";
const char* simple_jit = "torch/csrc/deploy/example/generated/simple_jit";

const char* path(const char* envname, const char* path) {
  const char* e = getenv(envname);
  return e ? e : path;
}

TEST(TorchpyTest, LoadLibrary) {
  torch::deploy::InterpreterManager m(1);
  torch::deploy::Package p = m.load_package(
      path("LOAD_LIBRARY", "torch/csrc/deploy/example/generated/load_library"));
  auto model = p.load_pickle("fn", "fn.pkl");
  model({});
}

TEST(TorchpyTest, InitTwice) {
  { torch::deploy::InterpreterManager m(2); }
  { torch::deploy::InterpreterManager m(1); }
}

TEST(TorchpyTest, DifferentInterps) {
  torch::deploy::InterpreterManager m(2);
  m.register_module_source("check_none", "check = id(None)\n");
  int64_t id0 = 0, id1 = 0;
  {
    auto I = m.all_instances()[0].acquire_session();
    id0 = I.global("check_none", "check").toIValue().toInt();
  }
  {
    auto I = m.all_instances()[1].acquire_session();
    id1 = I.global("check_none", "check").toIValue().toInt();
  }
  ASSERT_NE(id0, id1);
}

TEST(TorchpyTest, SimpleModel) {
  compare_torchpy_jit(path("SIMPLE", simple), path("SIMPLE_JIT", simple_jit));
}

TEST(TorchpyTest, ResNet) {
  compare_torchpy_jit(
      path("RESNET", "torch/csrc/deploy/example/generated/resnet"),
      path("RESNET_JIT", "torch/csrc/deploy/example/generated/resnet_jit"));
}

TEST(TorchpyTest, Movable) {
  torch::deploy::InterpreterManager m(1);
  torch::deploy::ReplicatedObj obj;
  {
    auto I = m.acquire_one();
    auto model =
        I.global("torch.nn", "Module")(std::vector<torch::deploy::Obj>());
    obj = I.create_movable(model);
  }
  obj.acquire_session();
}

TEST(TorchpyTest, MultiSerialSimpleModel) {
  torch::deploy::InterpreterManager manager(3);
  torch::deploy::Package p = manager.load_package(path("SIMPLE", simple));
  auto model = p.load_pickle("model", "model.pkl");
  auto ref_model = torch::jit::load(path("SIMPLE_JIT", simple_jit));

  auto input = torch::ones({10, 20});
  size_t ninterp = 3;
  std::vector<at::Tensor> outputs;

  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  for (const auto i : c10::irange(ninterp)) {
    outputs.push_back(model({input.alias()}).toTensor());
  }

  // Generate reference
  auto ref_output = ref_model.forward({input.alias()}).toTensor();

  // Compare all to reference
  for (const auto i : c10::irange(ninterp)) {
    ASSERT_TRUE(ref_output.equal(outputs[i]));
  }

  // test kwargs api with args
  std::vector<c10::IValue> args;
  args.emplace_back(input);
  std::unordered_map<std::string, c10::IValue> kwargs_empty;
  auto jit_output_args = model.call_kwargs(args, kwargs_empty).toTensor();
  ASSERT_TRUE(ref_output.equal(jit_output_args));

  // and with kwargs only
  std::unordered_map<std::string, c10::IValue> kwargs;
  kwargs["input"] = input;
  auto jit_output_kwargs = model.call_kwargs(kwargs).toTensor();
  ASSERT_TRUE(ref_output.equal(jit_output_kwargs));

  // test hasattr
  ASSERT_TRUE(model.hasattr("forward"));
  ASSERT_FALSE(model.hasattr("make_prediction"));
}

TEST(TorchpyTest, ThreadedSimpleModel) {
  size_t nthreads = 3;
  torch::deploy::InterpreterManager manager(nthreads);

  torch::deploy::Package p = manager.load_package(path("SIMPLE", simple));
  auto model = p.load_pickle("model", "model.pkl");
  auto ref_model = torch::jit::load(path("SIMPLE_JIT", simple_jit));

  auto input = torch::ones({10, 20});

  std::vector<at::Tensor> outputs;

  std::vector<std::future<at::Tensor>> futures;
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  for (const auto i : c10::irange(nthreads)) {
    futures.push_back(std::async(std::launch::async, [&model]() {
      auto input = torch::ones({10, 20});
      for (const auto i : c10::irange(100)) {
        model({input.alias()}).toTensor();
      }
      auto result = model({input.alias()}).toTensor();
      return result;
    }));
  }
  for (const auto i : c10::irange(nthreads)) {
    outputs.push_back(futures[i].get());
  }

  // Generate reference
  auto ref_output = ref_model.forward({input.alias()}).toTensor();

  // Compare all to reference
  for (const auto i : c10::irange(nthreads)) {
    ASSERT_TRUE(ref_output.equal(outputs[i]));
  }
}

TEST(TorchpyTest, ThrowsSafely) {
  // See explanation in deploy.h
  torch::deploy::InterpreterManager manager(3);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW(manager.load_package("some garbage path"), c10::Error);

  torch::deploy::Package p = manager.load_package(path("SIMPLE", simple));
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW(p.load_pickle("some other", "garbage path"), c10::Error);

  auto model = p.load_pickle("model", "model.pkl");
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW(model(at::IValue("unexpected input")), c10::Error);
}

TEST(TorchpyTest, AcquireMultipleSessionsInTheSamePackage) {
  torch::deploy::InterpreterManager m(1);

  torch::deploy::Package p = m.load_package(path("SIMPLE", simple));
  auto I = p.acquire_session();

  auto I1 = p.acquire_session();
}

TEST(TorchpyTest, AcquireMultipleSessionsInDifferentPackages) {
  torch::deploy::InterpreterManager m(1);

  torch::deploy::Package p = m.load_package(path("SIMPLE", simple));
  auto I = p.acquire_session();

  torch::deploy::Package p1 = m.load_package(
      path("RESNET", "torch/csrc/deploy/example/generated/resnet"));
  auto I1 = p1.acquire_session();
}

TEST(TorchpyTest, TensorSharingNotAllowed) {
  size_t nthreads = 2;
  torch::deploy::InterpreterManager m(nthreads);
  // generate a tensor from one interpreter
  auto I0 = m.all_instances()[0].acquire_session();
  auto I1 = m.all_instances()[1].acquire_session();
  auto obj = I0.global("torch", "empty")({I0.from_ivalue(2)});
  auto t = obj.toIValue().toTensor();
  // try to feed it to the other interpreter, should error
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_THROW(I1.global("torch", "sigmoid")({t}), c10::Error);
}

TEST(TorchpyTest, TaggingRace) {
  // At time of writing, this takes about 7s to run on DEBUG=1.  I think
  // this is OK, but feel free to fiddle with the knobs here to reduce the
  // runtime
  constexpr int64_t trials = 4;
  constexpr int64_t nthreads = 16;
  torch::deploy::InterpreterManager m(nthreads);
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  for (const auto n : c10::irange(trials)) {
    at::Tensor t = torch::empty(2);
    std::atomic<int64_t> success(0);
    std::atomic<int64_t> failed(0);
    at::parallel_for(0, nthreads, 1, [&](int64_t begin, int64_t end) {
      for (const auto i : c10::irange(begin, end)) {
        auto I = m.all_instances()[i].acquire_session();
        try {
          I.from_ivalue(t);
          success++;
        } catch (const c10::Error& e) {
          failed++;
        }
      }
    });
    ASSERT_EQ(success, 1);
    ASSERT_EQ(failed, nthreads - 1);
  }
}

TEST(TorchpyTest, DisarmHook) {
  at::Tensor t = torch::empty(2);
  {
    torch::deploy::InterpreterManager m(1);
    auto I = m.acquire_one();
    I.from_ivalue(t);
  } // unload the old interpreter
  torch::deploy::InterpreterManager m(1);
  auto I = m.acquire_one();
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_THROW(I.from_ivalue(t), c10::Error); // NOT a segfault
}

TEST(TorchpyTest, RegisterModule) {
  torch::deploy::InterpreterManager m(2);
  m.register_module_source("foomodule", "def add1(x): return x + 1\n");
  for (const auto& interp : m.all_instances()) {
    auto I = interp.acquire_session();
    AT_ASSERT(3 == I.global("foomodule", "add1")({2}).toIValue().toInt());
  }
}

TEST(TorchpyTest, FxModule) {
  size_t nthreads = 3;
  torch::deploy::InterpreterManager manager(nthreads);
  torch::deploy::Package p = manager.load_package(path(
      "SIMPLE_LEAF_FX", "torch/csrc/deploy/example/generated/simple_leaf_fx"));
  auto model = p.load_pickle("model", "model.pkl");

  std::vector<at::Tensor> outputs;
  auto input = torch::ones({5, 10});
  for (const auto i : c10::irange(nthreads)) {
    outputs.push_back(model({input.alias()}).toTensor());
  }

  // reference model
  auto ref_model = torch::jit::load(path(
      "SIMPLE_LEAF_JIT",
      "torch/csrc/deploy/example/generated/simple_leaf_jit"));

  auto ref_output = ref_model.forward({input.alias()}).toTensor();

  // Compare all to reference
  for (const auto i : c10::irange(nthreads)) {
    ASSERT_TRUE(ref_output.equal(outputs[i]));
  }
}

#ifdef TEST_CUSTOM_LIBRARY
thread_local int in_another_module = 5;
TEST(TorchpyTest, SharedLibraryLoad) {
  torch::deploy::InterpreterManager manager(2);
  auto no_args = at::ArrayRef<torch::deploy::Obj>();
  for (auto& interp : manager.all_instances()) {
    auto I = interp.acquire_session();

    const char* test_lib_path = getenv("LIBTEST_DEPLOY_LIB");
    if (!test_lib_path) {
      I.global("sys", "path").attr("append")({"torch/csrc/deploy"});
      I.global("test_deploy_python", "setup")({getenv("PATH")});
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
      char buf[PATH_MAX];
      strncpy(buf, test_lib_path, PATH_MAX);
      dirname(buf);
      I.global("sys", "path").attr("append")({buf});
    }

    AT_ASSERT(I.global("libtest_deploy_lib", "check_initial_state")(no_args)
                  .toIValue()
                  .toBool());
    ASSERT_TRUE(
        I.global("libtest_deploy_lib", "simple_add")({5, 4})
            .toIValue()
            .toInt() == 9);
    // I.global("numpy", "array"); // force numpy to load here so it is loaded
    //                             // twice before we run the tests
  }
  for (auto& interp : manager.all_instances()) {
    auto I = interp.acquire_session();
    // auto i =
    //     I.global("test_deploy_python", "numpy_test")({1}).toIValue().toInt();
    I.global("libtest_deploy_lib", "raise_and_catch_exception")({true});
    try {
      I.global("libtest_deploy_lib", "raise_exception")(no_args);
      ASSERT_TRUE(false); // raise_exception did not throw?
    } catch (std::exception& err) {
      ASSERT_TRUE(std::string(err.what()).find("yet") != std::string::npos);
    }
    in_another_module = 6;
    ASSERT_TRUE(
        I.global("libtest_deploy_lib", "get_in_another_module")(no_args)
            .toIValue()
            .toInt() == 6);
    ASSERT_TRUE(
        I.global("libtest_deploy_lib", "get_bar")(no_args).toIValue().toInt() ==
        14);
    {
      std::thread foo([&] {
        I.global("libtest_deploy_lib", "set_bar")({13});
        ASSERT_TRUE(
            I.global("libtest_deploy_lib", "get_bar")(no_args)
                .toIValue()
                .toInt() == 13);
      });
      foo.join();
    }
    ASSERT_TRUE(
        I.global("libtest_deploy_lib", "get_bar_destructed")(no_args)
            .toIValue()
            .toInt() == 1);
    I.global("libtest_deploy_lib", "set_bar")({12});
  }
}
#endif

TEST(TorchpyTest, UsesDistributed) {
  const auto model_filename = path(
      "USES_DISTRIBUTED",
      "torch/csrc/deploy/example/generated/uses_distributed");
  torch::deploy::InterpreterManager m(1);
  torch::deploy::Package p = m.load_package(model_filename);
  {
    auto I = p.acquire_session();
    I.self.attr("import_module")({"uses_distributed"});
  }
}

TEST(TorchpyTest, Autograd) {
  torch::deploy::InterpreterManager m(2);
  m.register_module_source("autograd_test", R"PYTHON(
import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
loss.backward()
# result = w.grad
result = torch.Tensor([1,2,3])
)PYTHON");
  at::Tensor w_grad0, w_grad1;
  {
    auto I = m.all_instances()[0].acquire_session();
    w_grad0 = I.global("autograd_test", "result").toIValue().toTensor();
  }
  {
    auto I = m.all_instances()[1].acquire_session();
    w_grad1 = I.global("autograd_test", "result").toIValue().toTensor();
  }
  EXPECT_TRUE(w_grad0.equal(w_grad1));
}
