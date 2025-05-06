在 Python 中，函数修改外部变量的方式与 C++ 的引用传递 (`&`) 有所不同。Python 使用的是“传递对象引用”（pass-by-object-reference）的方式。这意味着函数接收的是外部对象的引用（内存地址），但能否直接修改外部变量取决于该对象的类型（可变还是不可变）。

以下是几种在 Python 函数中影响或修改外部变量的方法：

1. **直接修改可变对象 (Mutable Objects)**：
    
    如果外部变量是可变类型（如列表 list、字典 dict、集合 set），函数可以通过接收到的引用直接修改对象的内容，这个修改会反映到外部变量上。这最接近 C++ 引用的效果。
    
    
    
    ```python
    def modify_list(my_list):
      # 函数接收列表的引用
      my_list.append(4) # 直接修改列表对象
      print("函数内部:", my_list)
    
    # 外部变量 (列表是可变对象)
    external_list = [1, 2, 3]
    print("函数调用前:", external_list)
    modify_list(external_list)
    print("函数调用后:", external_list) # 外部列表也被修改了
    ```
    
    输出：
    
    ```
    函数调用前: [1, 2, 3]
    函数内部: [1, 2, 3, 4]
    函数调用后: [1, 2, 3, 4]
    ```
    
2. **返回新值并重新赋值 (针对不可变对象)**：
    
    如果外部变量是不可变类型（如数字 int, float, 字符串 str, 元组 tuple），函数无法直接修改原始对象。函数内部对参数的任何重新赋值只会创建一个新的局部对象。要“修改”外部的不可变变量，函数应该 return 新的值，然后在函数外部将返回的值重新赋给原始变量。
    
    
    
    ```python
    def modify_number(num):
      # num 接收外部数字的引用，但数字是不可变的
      num = num + 10 # 这里创建了一个新的局部变量 num，外部的 x 不受影响
      print("函数内部 num:", num)
      return num # 返回计算后的新值
    
    # 外部变量 (数字是不可变对象)
    x = 5
    print("函数调用前 x:", x)
    new_x = modify_number(x) # 调用函数并接收返回值
    print("函数调用后 x (未改变):", x)
    print("函数返回值 new_x:", new_x)
    x = new_x # 将返回值重新赋给 x，这才“修改”了 x
    print("重新赋值后 x:", x)
    ```
    
    输出：
    
    ```
    函数调用前 x: 5
    函数内部 num: 15
    函数调用后 x (未改变): 5
    函数返回值 new_x: 15
    重新赋值后 x: 15
    ```
    
3. **使用 global 关键字 (不推荐)**：
    
    你可以使用 global 关键字明确声明函数内部要修改的是全局作用域中的变量。这能直接修改全局变量，但通常不推荐这样做，因为它会破坏函数的封装性，使代码难以理解和维护。
    
    
    
    ```python
    count = 0 # 全局变量
    
    def increment_global_count():
      global count # 声明要修改全局变量 count
      count += 1
      print("函数内部 count:", count)
    
    print("调用前 count:", count)
    increment_global_count()
    print("调用后 count:", count) # 全局变量被修改
    ```
    
    输出：
    
    ```
    调用前 count: 0
    函数内部 count: 1
    调用后 count: 1
    ```
    
4. **使用 nonlocal 关键字 (用于嵌套函数)**：
    
    在嵌套函数中，如果想修改外层（但非全局）函数的变量，可以使用 nonlocal 关键字。
    
    
    
    ```python
    def outer_function():
      level = "outer"
      def inner_function():
        nonlocal level # 声明要修改外层函数的 level
        level = "inner"
        print("Inner:", level)
      print("Outer (before):", level)
      inner_function()
      print("Outer (after):", level) # 外层变量被修改
    
    outer_function()
    ```
    
    输出：
    
    ```
    Outer (before): outer
    Inner: inner
    Outer (after): inner
    ```
    

**总结：**

- 修改**可变对象**（列表、字典等）：直接在函数内部操作即可，效果类似 C++ 引用。
- 修改**不可变对象**（数字、字符串等）：函数**返回**新值，调用者**重新赋值**给原变量。这是 Pythonic 的做法。
- 直接修改**全局变量**：使用 `global`，但不推荐。
- 直接修改**嵌套函数的外部变量**：使用 `nonlocal`。

对于大多数情况，推荐优先使用方法 1（修改可变对象）和方法 2（返回新值）。