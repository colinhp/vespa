// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/eval/eval/simple_value.h>
#include <vespa/eval/eval/test/tensor_model.hpp>
#include <vespa/vespalib/gtest/gtest.h>

using namespace vespalib::eval;
using namespace vespalib::eval::test;

std::vector<Layout> layouts = {
    {},
    {x(3)},
    {x(3),y(5)},
    {x(3),y(5),z(7)},
    float_cells({x(3),y(5),z(7)}),
    {x({"a","b","c"})},
    {x({"a","b","c"}),y({"foo","bar"})},
    {x({"a","b","c"}),y({"foo","bar"}),z({"i","j","k","l"})},
    float_cells({x({"a","b","c"}),y({"foo","bar"}),z({"i","j","k","l"})}),
    {x(3),y({"foo", "bar"}),z(7)},
    {x({"a","b","c"}),y(5),z({"i","j","k","l"})},
    float_cells({x({"a","b","c"}),y(5),z({"i","j","k","l"})})
};

TEST(SimpleValueTest, simple_values_can_be_converted_from_and_to_tensor_spec) {
    for (const auto &layout: layouts) {
        TensorSpec expect = spec(layout, N());
        std::unique_ptr<NewValue> value = new_value_from_spec(expect, SimpleValueBuilderFactory());
        TensorSpec actual = spec_from_new_value(*value);
        EXPECT_EQ(actual, expect);
    }
}

TEST(SimpleValueTest, simple_value_can_be_built_and_inspected) {
    ValueType type = ValueType::from_spec("tensor<float>(x{},y[2],z{})");
    SimpleValueBuilderFactory factory;
    std::unique_ptr<ValueBuilder<float>> builder = factory.create_value_builder<float>(type);
    float seq = 0.0;
    for (vespalib::string x: {"a", "b", "c"}) {
        for (vespalib::string y: {"aa", "bb"}) {
            auto subspace = builder->add_subspace({x, y});
            EXPECT_EQ(subspace.size(), 2);
            subspace[0] = seq + 1.0;
            subspace[1] = seq + 5.0;
            seq += 10.0;
        }
        seq += 100.0;
    }
    std::unique_ptr<NewValue> value = builder->build(std::move(builder));
    EXPECT_EQ(value->index().size(), 6);
    auto view = value->index().create_view({0});
    vespalib::stringref query = "b";
    vespalib::stringref label;
    size_t subspace;
    view->lookup({&query});
    EXPECT_TRUE(view->next_result({&label}, subspace));
    EXPECT_EQ(label, "aa");
    EXPECT_EQ(subspace, 2);
    EXPECT_TRUE(view->next_result({&label}, subspace));
    EXPECT_EQ(label, "bb");
    EXPECT_EQ(subspace, 3);
    EXPECT_FALSE(view->next_result({&label}, subspace));
}

TEST(SimpleValueTest, join_traverse_plan_can_be_created) {
    auto lhs = ValueType::from_spec("tensor(a{},b[6],c[5],e[3],f[2],g{})");
    auto rhs = ValueType::from_spec("tensor(a{},b[6],c[5],d[4],h{})");
    auto plan = JoinTraversePlan(lhs, rhs);
    std::vector<size_t> expect_loop = {30,4,6};
    std::vector<size_t> expect_lhs_stride = {6,0,1};
    std::vector<size_t> expect_rhs_stride = {4,1,0};
    EXPECT_EQ(plan.loop_cnt, expect_loop);
    EXPECT_EQ(plan.lhs_stride, expect_lhs_stride);
    EXPECT_EQ(plan.rhs_stride, expect_rhs_stride);
}

TEST(SimpleValueTest, join_mapped_overlap_can_be_created) {
    auto lhs = ValueType::from_spec("tensor(a{},b[6],c[5],e[3],f[2],g{})");
    auto rhs = ValueType::from_spec("tensor(a{},b[6],c[5],d[4],h{})");
    auto overlap = JoinMappedOverlap(lhs, rhs);
    using SRC = JoinMappedOverlap::Source;
    std::vector<SRC> expect_sources = {SRC::BOTH,SRC::LHS,SRC::RHS};
    EXPECT_EQ(overlap.sources, expect_sources);
}

GTEST_MAIN_RUN_ALL_TESTS()
