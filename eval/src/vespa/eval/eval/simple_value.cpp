// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "simple_value.h"
#include "tensor_spec.h"
#include "inline_operation.h"
#include <vespa/vespalib/util/typify.h>
#include <vespa/vespalib/util/visit_ranges.h>
#include <vespa/vespalib/util/overload.h>

#include <vespa/log/log.h>
LOG_SETUP(".eval.simple_value");

namespace vespalib::eval {

//-----------------------------------------------------------------------------

namespace {

struct CreateSimpleValueBuilderBase {
    template <typename T> static std::unique_ptr<ValueBuilderBase> invoke(const ValueType &type,
            size_t num_mapped_in, size_t subspace_size_in)
    {
        assert(check_cell_type<T>(type.cell_type()));
        return std::make_unique<SimpleValueT<T>>(type, num_mapped_in, subspace_size_in);
    }
};

struct CreateValueFromTensorSpec {
    template <typename T> static std::unique_ptr<NewValue> invoke(const ValueType &type, const TensorSpec &spec, const ValueBuilderFactory &factory) {
        using SparseKey = std::vector<vespalib::stringref>;
        using DenseMap = std::map<size_t,T>;
        std::map<SparseKey,DenseMap> map;
        for (const auto &entry: spec.cells()) {
            SparseKey sparse_key;
            size_t dense_key = 0;
            for (const auto &dim: type.dimensions()) {
                auto pos = entry.first.find(dim.name);
                assert(pos != entry.first.end());
                assert(pos->second.is_mapped() == dim.is_mapped());
                if (dim.is_mapped()) {
                    sparse_key.emplace_back(pos->second.name);
                } else {
                    dense_key = (dense_key * dim.size) + pos->second.index;
                }
            }
            map[sparse_key][dense_key] = entry.second;
        }
        auto builder = factory.create_value_builder<T>(type, type.count_mapped_dimensions(), type.dense_subspace_size(), map.size());
        for (const auto &entry: map) {
            auto subspace = builder->add_subspace(entry.first);
            for (const auto &cell: entry.second) {
                subspace[cell.first] = cell.second;
            }
        }
        return builder->build(std::move(builder));
    }
};

struct CreateTensorSpecFromValue {
    template <typename T> static TensorSpec invoke(const NewValue &value) {
        auto cells = value.cells().typify<T>();
        TensorSpec spec(value.type().to_spec());
        size_t subspace_id = 0;
        size_t subspace_size = value.type().dense_subspace_size();
        std::vector<vespalib::stringref> labels(value.type().count_mapped_dimensions());
        std::vector<vespalib::stringref*> label_refs;
        for (auto &label: labels) {
            label_refs.push_back(&label);
        }
        auto view = value.index().create_view({});
        view->lookup({});
        while (view->next_result(label_refs, subspace_id)) {
            size_t label_idx = 0;
            TensorSpec::Address addr;
            for (const auto &dim: value.type().dimensions()) {
                if (dim.is_mapped()) {
                    addr.emplace(dim.name, labels[label_idx++]);
                }
            }
            for (size_t i = 0; i < subspace_size; ++i) {
                size_t dense_key = i;
                for (auto dim = value.type().dimensions().rbegin();
                     dim != value.type().dimensions().rend(); ++dim)
                {
                    if (dim->is_indexed()) {
                        size_t label = dense_key % dim->size;
                        addr.emplace(dim->name, label).first->second = TensorSpec::Label(label);
                        dense_key /= dim->size;
                    }
                }
                spec.add(addr, cells[(subspace_size * subspace_id) + i]);
            }
        }
        return spec;
    }
};

class SimpleValueView : public NewValue::Index::View {
private:
    using Addr = std::vector<vespalib::string>;
    using Map = std::map<Addr,size_t>;
    using Itr = Map::const_iterator;

    const Map          &_index;
    size_t              _num_mapped;
    std::vector<size_t> _match_dims;
    std::vector<size_t> _extract_dims;
    Addr                _query;
    Itr                 _pos;

    bool is_direct_lookup() const { return (_match_dims.size() == _num_mapped); }
    bool is_match() const {
        assert(_pos->first.size() == _num_mapped);
        for (size_t idx: _match_dims) {
            if (_query[idx] != _pos->first[idx]) {
                return false;
            }
        }
        return true;
    }

public:
    SimpleValueView(const Map &index, const std::vector<size_t> &match_dims, size_t num_mapped)
        : _index(index), _num_mapped(num_mapped), _match_dims(match_dims), _extract_dims(), _query(num_mapped, ""), _pos(_index.end())
    {
        auto pos = _match_dims.begin();
        for (size_t i = 0; i < _num_mapped; ++i) {
            if ((pos == _match_dims.end()) || (*pos != i)) {
                _extract_dims.push_back(i);
            } else {
                ++pos;
            }
        }
        assert(pos == _match_dims.end());
        assert((_match_dims.size() + _extract_dims.size()) == _num_mapped);
    }

    void lookup(const std::vector<const vespalib::stringref*> &addr) override {
        assert(addr.size() == _match_dims.size());
        for (size_t i = 0; i < _match_dims.size(); ++i) {
            _query[_match_dims[i]] = *addr[i];
        }
        if (is_direct_lookup()) {
            _pos = _index.find(_query);
        } else {
            _pos = _index.begin();
        }
    }

    bool next_result(const std::vector<vespalib::stringref*> &addr_out, size_t &idx_out) override {
        assert(addr_out.size() == _extract_dims.size());
        while (_pos != _index.end()) {
            if (is_match()) {
                for (size_t i = 0; i < _extract_dims.size(); ++i) {
                    *addr_out[i] = _pos->first[_extract_dims[i]];
                }
                idx_out = _pos->second;
                if (is_direct_lookup()) {
                    _pos = _index.end();
                } else {
                    ++_pos;
                }
                return true;
            }
            ++_pos;
        }
        return false;
    }
};

// Treats all values as mixed tensors. Needs output cell type as well
// as input cell types since output cell type cannot always be
// directly inferred.

struct GenericJoin {
    template <typename LCT, typename RCT, typename OCT, typename Fun> static std::unique_ptr<NewValue>
    invoke(const NewValue &a, const NewValue &b, join_fun_t function,
           const JoinTraversePlan &plan, const ValueType &res_type, const ValueBuilderFactory &factory)
    {
        Fun fun(function);
        auto builder = factory.create_value_builder<OCT>(res_type);
        bool swap = (b.index().size() < a.index().size());
        (void) fun;
        (void) swap;
        (void) a;
        (void) b;
        (void) plan;
        return builder->build(std::move(builder));
    }
};

}

//-----------------------------------------------------------------------------

void
SimpleValue::add_mapping(const std::vector<vespalib::stringref> &addr)
{
    size_t id = _index.size();
    std::vector<vespalib::string> my_addr;
    for (const auto &label: addr) {
        my_addr.push_back(label);
    }
    auto res = _index.emplace(std::move(my_addr), id);
    assert(res.second);
}

SimpleValue::SimpleValue(const ValueType &type, size_t num_mapped_in, size_t subspace_size_in)
    : _type(type),
      _num_mapped(num_mapped_in),
      _subspace_size(subspace_size_in),
      _index()
{
    assert(_type.count_mapped_dimensions() == _num_mapped);
    assert(_type.dense_subspace_size() == _subspace_size);
}

SimpleValue::~SimpleValue() = default;

std::unique_ptr<NewValue::Index::View>
SimpleValue::create_view(const std::vector<size_t> &dims) const
{
    return std::make_unique<SimpleValueView>(_index, dims, _num_mapped);
}

//-----------------------------------------------------------------------------

template <typename T>
SimpleValueT<T>::SimpleValueT(const ValueType &type, size_t num_mapped_in, size_t subspace_size_in)
    : SimpleValue(type, num_mapped_in, subspace_size_in),
      _cells()
{
}

template <typename T>
SimpleValueT<T>::~SimpleValueT() = default;

template <typename T>
ArrayRef<T>
SimpleValueT<T>::add_subspace(const std::vector<vespalib::stringref> &addr)
{
    size_t old_size = _cells.size();
    assert(old_size == (index().size() * subspace_size()));
    add_mapping(addr);
    _cells.resize(old_size + subspace_size());
    return ArrayRef<T>(&_cells[old_size], subspace_size());
}

//-----------------------------------------------------------------------------

std::unique_ptr<ValueBuilderBase>
SimpleValueBuilderFactory::create_value_builder_base(const ValueType &type,
                                                     size_t num_mapped_in, size_t subspace_size_in, size_t) const
{
    return typify_invoke<1,TypifyCellType,CreateSimpleValueBuilderBase>(type.cell_type(), type, num_mapped_in, subspace_size_in);
}

//-----------------------------------------------------------------------------

JoinTraversePlan::JoinTraversePlan(const ValueType &lhs_type, const ValueType &rhs_type)
    : loop_cnt(), lhs_stride(), rhs_stride()
{
    enum class Case { NONE, LHS, RHS, BOTH };
    Case prev_case = Case::NONE;
    auto update_plan = [&](Case my_case, size_t my_size, size_t in_lhs, size_t in_rhs) {
        if (my_case == prev_case) {
            assert(!loop_cnt.empty());
            loop_cnt.back() *= my_size;
        } else {
            loop_cnt.push_back(my_size);
            lhs_stride.push_back(in_lhs);
            rhs_stride.push_back(in_rhs);
            prev_case = my_case;
        }
    };
    auto visitor = overload
                   {
                       [&](visit_ranges_first, const auto &a) { update_plan(Case::LHS, a.size, 1, 0); },
                       [&](visit_ranges_second, const auto &b) { update_plan(Case::RHS, b.size, 0, 1); },
                       [&](visit_ranges_both, const auto &a, const auto &) { update_plan(Case::BOTH, a.size, 1, 1); }
                   };
    auto lhs_dims = lhs_type.nontrivial_indexed_dimensions();
    auto rhs_dims = rhs_type.nontrivial_indexed_dimensions();
    visit_ranges(visitor, lhs_dims.begin(), lhs_dims.end(), rhs_dims.begin(), rhs_dims.end(),
                 [](const auto &a, const auto &b){ return (a.name < b.name); });
    size_t lhs_size = 1;
    size_t rhs_size = 1;
    for (size_t i = loop_cnt.size(); i-- > 0; ) {
        if (lhs_stride[i]) {
            lhs_stride[i] = lhs_size;
            lhs_size *= loop_cnt[i];
        }
        if (rhs_stride[i]) {
            rhs_stride[i] = rhs_size;
            rhs_size *= loop_cnt[i];
        }
    }
}

//-----------------------------------------------------------------------------

JoinMappedOverlap::JoinMappedOverlap(const ValueType &lhs_type, const ValueType &rhs_type)
    : sources()
{
    auto visitor = overload
                   {
                       [&](visit_ranges_first, const auto &) { sources.push_back(Source::LHS); },
                       [&](visit_ranges_second, const auto &) { sources.push_back(Source::RHS); },
                       [&](visit_ranges_both, const auto &, const auto &) { sources.push_back(Source::BOTH); }
                   };
    auto lhs_dims = lhs_type.mapped_dimensions();
    auto rhs_dims = rhs_type.mapped_dimensions();
    visit_ranges(visitor, lhs_dims.begin(), lhs_dims.end(), rhs_dims.begin(), rhs_dims.end(),
                 [](const auto &a, const auto &b){ return (a.name < b.name); });
}

//-----------------------------------------------------------------------------

using JoinTypify = TypifyValue<TypifyCellType,operation::TypifyOp2>;

std::unique_ptr<NewValue> new_join(const NewValue &a, const NewValue &b, join_fun_t function, const ValueBuilderFactory &factory) {
    auto res_type = ValueType::join(a.type(), b.type());
    JoinTraversePlan plan(a.type(), b.type());
    return typify_invoke<4,JoinTypify,GenericJoin>(a.type().cell_type(), b.type().cell_type(), res_type.cell_type(), function,
                                                   a, b, function, plan, res_type, factory);
}

//-----------------------------------------------------------------------------

std::unique_ptr<NewValue> new_value_from_spec(const TensorSpec &spec, const ValueBuilderFactory &factory) {
    ValueType type = ValueType::from_spec(spec.type());
    assert(!type.is_error());
    return typify_invoke<1,TypifyCellType,CreateValueFromTensorSpec>(type.cell_type(), type, spec, factory);
}

//-----------------------------------------------------------------------------

TensorSpec spec_from_new_value(const NewValue &value) {
    return typify_invoke<1,TypifyCellType,CreateTensorSpecFromValue>(value.type().cell_type(), value);
}

//-----------------------------------------------------------------------------

}
