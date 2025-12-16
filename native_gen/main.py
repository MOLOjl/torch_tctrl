r"""
生成at::native函数的warpper函数
有些额外需要单独处理的函数并不能覆盖，比如返回标量的函数，返回值为void的函数
void checkpoint__foreach_mul_(at::TensorList self, const at::Tensor & other)
Scalar checkpoint__local_scalar_dense(at::Tensor const& a)
OptionalIntArrayRef的参数也需要额外处理
"""
import yaml
from processer import list_head_files, get_mapped_op_head_files, get_to_be_impl_ops, logger
import re


def export_to_be_impl_ops():
    with open('./Llama2-op-list.txt', 'r') as f:
        data = f.readlines()
        op_list = []
        for row in data:
            if 'aten::' in row:
                op_list.append(row.replace('\n', ''))
        del data
    # print(op_list)

    with open('./cur_native_functions.yaml', 'r') as f:
        func_mapping = yaml.load(f, Loader=yaml.FullLoader)

    to_impl_aten_op_mapping = []
    have_recored = []
    not_match_ops = []

    def add_check(op_name, func_dict, have_recored):
        func_name = str(func_dict['func']).split('.')[0] if '.' in func_dict['func'] else str(func_dict['func']).split('(')[0]

        if op_name not in func_name:
            return False
        elif func_dict['func'] in have_recored:
            return False
        
        if 'mps' in func_name or 'miopen' in func_name or 'mkldnn' in func_name or 'sparse' in func_name or 'foreach' in func_name:
            return False

        if 'dispatch' not in func_dict.keys():
            return False
        # elif ('CUDA' in func_dict['dispatch'].keys()) or ('CPU, CUDA' in func_dict['dispatch'].keys()):
        #     return True
        # else:
        #     return False
        return op_name == func_name
    
    counts = 0
    for li in op_list:
        op_name = str(li).split("::")[1]
        for di in func_mapping:
            if add_check(op_name, di, have_recored):
                to_impl_aten_op_mapping.append(di)
                logger(op_name, di)
                have_recored.append(dict(di)['func'])
        if len(to_impl_aten_op_mapping) > counts:
            counts = len(to_impl_aten_op_mapping)
        else:
            not_match_ops.append(li)

    for row in not_match_ops:
        logger(row)
    logger('to impl ops:', len(to_impl_aten_op_mapping), 'not match ops:', len(not_match_ops),'/',len(op_list))
    # with open('./to_be_impled.yaml', 'w') as f:
    #     yaml.dump(to_impl_aten_op_mapping, f, sort_keys=False)
# print(to_impl_aten_op_mapping)
 
def gen_new_func(func_sign):
    r"""
    1. 一般类型:
    Tensor checkpoint_where(at::Tensor const& a, at::Tensor const& b, at::Tensor const& c) {
        rematerialize_function_t rt =
            [=](const Tensors& vec) -> Tensors {
            return {at::where(vec.at(0), vec.at(1), vec.at(2))};
            };
        return CheckpointTensorImpl::make("where", rt, {a, b, c})[0];
    }

    2. optional参数的处理:
    c10::MaybeOwned<Tensor> c_maybe_owned = at::borrow_from_optional_tensor(c);
    const Tensor& c_ = *c_maybe_owned;

    3. 非const的参数需要额外处理(其中的loss):
    Tensor& checkpoint_binary_cross_entropy_out(const Tensor& input, const Tensor& target, const c10::optional<Tensor>& weight_opt, int64_t reduction, Tensor& loss) {
        mutate_function_t mt =
            [=](const Tensors& vec) {
            Tensor input = vec.at(0), loss = vec.at(3);
            at::binary_cross_entropy_outf(input, vec.at(1), vec.at(2), reduction, loss);
            };
        c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
        const Tensor& w_ = *weight_maybe_owned;
        CheckpointTensorImpl::mutate("binary_cross_entropy_out", mt, {input, target, w_, loss}, {0});
        return loss;
    }

    4. TensorList的处理
    void _foreach_addcmul_(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars)

    5. SymIntArrayRef的处理
    """
    decl_name, return_type, org_func_name, params_str = func_sign
    tmp = params_str[1:-1].split(', ')
    # scalar代指tensor以外的类型
    tensor_params = []
    not_const_tensor_params = [] # at::Tensor& 需要特殊处理
    scalar_params = []
    optional_tensor_params = []
    optional_scalar_params = []
    arrayref_params = []          # ArrayRef需要额外拷贝处理
    tensorList_params = []       # TensorList数目较少，额外处理逻辑较多
    # 保持有序方便传参
    params_ordered = []
    for t in tmp:
        last_space_index = t.rfind(' ')
        p_type = t[:last_space_index]
        p_name = t[last_space_index+1:] 
        p_default = None
        if '=' in p_name:
            p_name, p_default = p_name.split('=')[0], p_name.split('=')[1]
        to_add = (p_type, p_name, p_default)
        params_ordered.append(to_add)
        # 这里要注意判断的优先级问题，可以有更简洁的写法，但是不重要
        if 'optional' in p_type and 'Tensor' in p_type:
            optional_tensor_params.append(to_add)
        elif 'TensorOptions' in p_type:
            scalar_params.append(to_add)
        elif 'TensorList' in p_type:
            tensorList_params.append(to_add)
        elif 'Tensor' in p_type and 'const' in p_type:
            tensor_params.append(to_add)
        elif 'Tensor' in p_type:
            not_const_tensor_params.append(to_add)
        elif 'ArrayRef' in p_type:
            arrayref_params.append(to_add)
        elif 'optional' in p_type:
            optional_scalar_params.append(to_add)
        else:
            scalar_params.append(to_add)
    
    def get_tensor_param_id(pd, params_ordered):
        pd_id = 0
        skip_id = 0
        for i in range(len(params_ordered)):
            if 'Tensor' not in params_ordered[i][0]: # 跳过非tensor
                skip_id += 1
            if pd == params_ordered[i]:
                pd_id = i - skip_id
                break
        return pd_id
    
    def clear_default_param_val(params_str):
        res = params_str
        while '=' in res:
            eq_char_pos = res.index('=')
            sep_char_pos = -1
            for i in range(eq_char_pos, len(res)):
                if res[i] == ',' or res[i] == ')':
                    sep_char_pos = i
                    break
            res = res[:eq_char_pos] + res[sep_char_pos:]
            # break
        return res

    def foreach_func_gen(params_ordered):
        func_str = '/// ' + str(func_sign) + '\n'
        func_str += return_type + ' checkpoint_' + org_func_name + clear_default_param_val(params_str) + ' {\n'
        for pd in tensorList_params:
            func_str += f'  Tensors {pd[1]}_;\n'
            func_str +=  "  for (const auto i : c10::irange("+pd[1]+".size())) {\n"
            func_str += f'    {pd[1]}_.push_back({pd[1]}[i].decheckpoint());\n'
            func_str +=  '  }\n'
        func_str += '  at::' + org_func_name + '('
        for pd in params_ordered:
            if pd in tensorList_params:
                func_str += 'at::TensorList(' + pd[1] + '_), '
            # elif pd in scalar_params:
            #     func_str += pd[1] + ', '
            elif pd in tensor_params:
                func_str += pd[1] + '.decheckpoint(), '
            else:
                func_str += pd[1] + ', '
        func_str = func_str[:-2]
        func_str += ');\n'
        func_str += '}\n'
        print(func_str)

    ### foreach函数生成
    if return_type == 'void' and 'foreach' in decl_name:
        foreach_func_gen(params_ordered)
        return

    ######## 生成函数 缩进是2个空格 #########
    func_str = '/// ' + str(func_sign) + '\n'
    if return_type != 'void':
        func_str += return_type + ' checkpoint_' + org_func_name + clear_default_param_val(params_str) + ' {\n'
    else:
        func_str += 'Tensor& checkpoint_' + org_func_name + clear_default_param_val(params_str) + ' {\n'
    
    ### 处理ref参数 ###
    for pd in arrayref_params:
        func_str += '  auto ' + pd[1] + '_ = ' + pd[1] + '.vec();\n'

    ### 生成rt函数 ###
    func_type = 'mt' if return_type == 'void' or org_func_name[-1] == '_' else 'rt'
    if func_type == 'mt': # rt 和 mt 的区分, 似乎torch2.1自己完成了mutation的转换 不需要手动转换(错误地 还是需要)
        func_str += '  mutate_function_t mt =\n'
        func_str += '    [=](const Tensors& vec) {\n'
    else:
        func_str += '  rematerialize_function_t rt =\n'
        func_str += '    [=](const Tensors& vec) -> Tensors {\n'

    ################# 生成rt内部代码 #################
    for pd in not_const_tensor_params: # 非const&的tensor需要单独赋值才行
        func_str += '      Tensor ' + pd[1] + ' = vec.at({});\n'.format(get_tensor_param_id(pd, params_ordered))
    
    ### 处理不同的返回值类型 ###
    if func_type == 'mt':
        func_str += '      at::' + org_func_name + '('
    elif 'tuple' in return_type:
        func_str += '      auto ret = at::' + org_func_name + '('
    else:
        func_str += '      return {at::' + org_func_name + '(' # TODO: rt 和 mt 的区分

    ### 传入调用at::xxx的参数 ###
    for pd in params_ordered:
        if pd in not_const_tensor_params:
            func_str += pd[1] + ', '
        elif pd in tensor_params or pd in optional_tensor_params:
            func_str += 'vec.at({}), '.format(get_tensor_param_id(pd, params_ordered))
        elif pd in arrayref_params:
            func_str += pd[1] + '_, '
        else:
            func_str += pd[1] + ', '
    func_str = func_str[:-2]

    ### rt内部返回值收尾 ###
    func_str += ')};\n' if func_type != 'mt' and 'tuple' not in return_type else ');\n'
    if 'tuple' in return_type:
        func_str += '      return {'
        for i in range(return_type.count('Tensor')):
            func_str += 'std::get<{}>(ret), '.format(i)
        func_str = func_str[:-2]
        func_str += '};\n'
    func_str += '    };\n'

    # 处理optional param
    for pd in optional_tensor_params:
        func_str += '  c10::MaybeOwned<Tensor> ' + pd[1] + '_maybe_owned = at::borrow_from_optional_tensor(' + pd[1] + ');\n'
        func_str += '  const Tensor& ' + pd[1] + '_ = *' + pd[1] + '_maybe_owned' + ';\n'
        
    # 返回值
    if func_type == 'mt':
        func_str += '  CheckpointTensorImpl::mutate(\"'+org_func_name+'\", mt, {'
        for pd in params_ordered:
            if pd in tensor_params or pd in not_const_tensor_params:
                func_str += pd[1] + ', '
            elif pd in optional_tensor_params:
                func_str += pd[1] + '_, '
        func_str = func_str[:-2] # 删除最后一个逗号
        func_str += '}, {0});\n'            # TODO: 后面的数组代表mutate操作对应参数的索引，需要修复
        func_str += '  return {out};\n'     # TODO: 需要找到应返回值的变量名
    else:
        if 'tuple' not in return_type:
            func_str += '  return CheckpointTensorImpl::make(\"'+decl_name+'\", rt, {'
            has_params_to_pass = False
            for pd in params_ordered:
                if pd in tensor_params or pd in not_const_tensor_params:
                    func_str += pd[1] + ', '
                    has_params_to_pass = True
                elif pd in optional_tensor_params:
                    func_str += pd[1] + '_, '
                    has_params_to_pass = True
            if has_params_to_pass:
                func_str = func_str[:-2] # 删除最后一个逗号
            func_str += '})[0];\n'
        else:
            func_str += '  auto ret = CheckpointTensorImpl::make(\"'+decl_name+'\", rt, {'
            has_tensor_params = False
            for pd in params_ordered:
                if pd in tensor_params or pd in not_const_tensor_params:
                    func_str += pd[1] + ', '
                    has_tensor_params = True
                elif pd in optional_tensor_params:
                    func_str += pd[1] + '_, '
                    has_tensor_params = True
            func_str = func_str[:-2] if has_tensor_params else func_str # 删除最后一个逗号
            func_str += '});\n'
            func_str += '  return {'
            for i in range(return_type.count('Tensor')):
                func_str += 'ret[{}], '.format(i)
            func_str = func_str[:-2]
            func_str += '};\n'

    func_str += '}\n'
    print(func_str)
    # logger(func_str)
    with open('./generated_func.txt', 'a') as f:
        f.write(func_str+'\n')

def register_in_yaml(func_sign, op_info, native_ops):
    decl_name, return_type, org_func_name, params_str = func_sign
    new_func = 'checkpoint_' + org_func_name
    register_func_name = op_info[-1]
    for op_reg in native_ops:
        if op_reg['func'][:op_reg['func'].index('(')] == decl_name[6:]:
            logger(decl_name, register_func_name, op_reg['func'])
            if 'dispatch' in op_reg.keys():
                op_reg['dispatch']['Checkpoint'] = new_func
                print(op_reg)
    # logger(new_func, op_info)

def get_func_signs_from_file(path, op_name, use_func, existed_list: list):
    with open(path, 'r') as f:
        data = f.readlines()
        code = ''.join(data)
    # ::std::tuple<at::Tensor,at::Tensor>
    r"""
    返回值类型
    at::Tensor
    ::std::tuple<at::Tensor,at::Tensor>
    ::std::vector<at::Tensor>
    at::Tensor &
    void

    inline at::Tensor & upsample_trilinear3d_out(at::Tensor & out, const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d=c10::nullopt, c10::optional<double> scales_h=c10::nullopt, c10::optional<double> scales_w=c10::nullopt)
    inline ::std::vector<at::Tensor> _foreach_lgamma(at::TensorList self) {
        return at::_ops::_foreach_lgamma::call(self);
    }
    特殊情况——结构体  暂时跳过(不直接用这种接口)
    struct TORCH_API structured_slow_conv_transpose2d_structured_cuda : public at::meta::structured_slow_conv_transpose2d {
        void impl(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::OptionalTensorRef bias, at::IntArrayRef stride, at::ArrayRef<int64_t> padding, at::ArrayRef<int64_t> output_padding, at::IntArrayRef dilation, const at::Tensor & out);
    };
    """
    logger('to find func:', use_func)
    # 首先检查有非结构体包装的情况
    # function_pattern = r"TORCH_API (\w*::\w*|\w*::\w* &|::std::tuple.*>|void) (\w*)(\(.*\));"
    # For an in-place variant of a native function (op name ends with an `_`) MUTATE
    function_pattern = r"(//.*)\ninline (\w*::\w*|\w*::\w* &|::std::tuple.*>|void|::std::vector.*>) (\w*)(\(.*\))"
    matches = re.findall(function_pattern, code, re.MULTILINE)
    logger(path)
    res = []
    # 重载全部函数
    for m in matches:
        if m not in existed_list:
            existed_list.append(m)
            decl_sign = m[0][3:str(m[0]).index('(')]
            return_type, func_name, params_str = m[1], m[2], m[3]
            res.append([decl_sign, return_type, func_name, params_str])
    # for m in matches: # 寻找严格匹配的
    #     native_func_name = m[2]
    #     if use_func == native_func_name:
    #         existed_list.append(m)
    #         return [m]
    # res = []
    # for m in matches: # 寻找严格匹配的, 这时是函数重载
    #     native_func_name = m[2]
    #     if op_name == native_func_name and m not in res and m not in existed_list:
    #         existed_list.append(m)
    #         res.append(m)
    if len(res) == 0:
        logger('no match func in:',path, op_name, use_func)
    return res

def all_func_sign_and_register():
    # export_to_be_impl_ops()
    ops = get_to_be_impl_ops()
    head_files = list_head_files()
    mapped_files = get_mapped_op_head_files(ops, head_files) # TODO: 处理Not match的函数
     
    count = 0
    native_ops = get_to_be_impl_ops('./cur_native_functions.yaml')
    not_match_list = []
    existed_list = []
    for row in mapped_files:
        # op_name, func_name, candidates_files[-1], op_dict['func']
        # logger(row)
        file_path = row[2]['path']
        op_name, func_name = row[0], row[1]
        func_signs = get_func_signs_from_file(file_path, op_name, func_name, existed_list)
        if len(func_signs) == 0:
            # logger('not match op', row)
            not_match_list.append(row)
            continue
        for func_sign in func_signs:
            logger(func_sign)
            if 'ArrayRef' in func_sign[-1]:
                gen_new_func(func_sign)
            # register_in_yaml(func_sign, row, native_ops)
        # count += 1
        # if count > 2:
        # break
    logger('------------------ not match ops, but seems like repetition --------------------')
    logger('no match func num:', len(not_match_list))
    for row in not_match_list:
        logger(row)
    # with open('./new_native_functions.yaml', 'w') as f:
    #     yaml.dump(native_ops, f, sort_keys=False)
    

def single_func_gen(fsign):
    func_head = fsign
    def split_func_sign(f_str):
        function_pattern = r"(\w*::\w*|\w*::\w* &|std::tuple.*>|void|std::vector.*>) (\w*)(\(.*\))"
        matches = re.findall(function_pattern, f_str.replace('\n', ''), re.MULTILINE)
        res = []
        for m in matches:
            return_type, func_name, params_str = m[0], m[1], m[2]
            decl_sign = 'aten::' + func_name
            res.append([decl_sign, return_type, func_name, params_str])
        return res[0]
    sign = split_func_sign(func_head)
    gen_new_func(sign)

def single_opsfile_func_gen(path):
    with open(path, 'r') as f:
        data = f.readlines()
        code = ''.join(data)
    # 首先检查有非结构体包装的情况
    # function_pattern = r"TORCH_API (\w*::\w*|\w*::\w* &|::std::tuple.*>|void) (\w*)(\(.*\));"
    # For an in-place variant of a native function (op name ends with an `_`) MUTATE
    function_pattern = r"(//.*)\ninline (\w*::\w*|\w*::\w* &|::std::tuple.*>|void|::std::vector.*>) (\w*)(\(.*\))"
    matches = re.findall(function_pattern, code, re.MULTILINE)
    logger(path)
    sign = []
    # 重载全部函数
    for m in matches:
        decl_sign = m[0][3:str(m[0]).index('(')]
        return_type, func_name, params_str = m[1], m[2], m[3]
        sign.append([decl_sign, return_type, func_name, params_str])
    for si in sign:
        gen_new_func(si)


#### TODO: 寻找native.h是错误的，需要直接找at::xxxx的函数即可，或者直接根据native_functions中的函数签名生成？
#### TODO: export_to_be_impl_ops有7个op是不太一样的，需要转换，对应.h中有10个func是不一样的
#### TODO: TensorList的转换
#### TODO: mutate的index选取还没修复，内容也有异常 + 某些ref可能还没处理

if __name__ == '__main__':
    # single_func_gen("""at::Tensor & normal_(double mean, double std, c10::optional<at::Generator> generator)""")
    single_func_gen("""at::Tensor logical_and(const at::Tensor & self, const at::Tensor & other)""")
    # single_opsfile_func_gen('/data/wangzehua/pytorch_dtb/torch/include/ATen/ops/pow.h')
    # all_func_sign_and_register()