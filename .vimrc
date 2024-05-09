vim9script

# compiler is named pylint, but runs all of pre-commit
compiler pylint_vim_env
nnoremap <buffer> <leader>l :update<CR>:vert Make<CR>
inoremap <buffer> <leader>l <esc>:update<CR>:vert Make<CR>

# run the main python file with
nnoremap <leader>m :update<CR>:ScratchTermReplaceU python src/posterize2/main.py<CR>
inoremap <leader>m <esc>:update<CR>:ScratchTermReplaceU python src/posterize2/main.py<CR>

# resize the code window after running a python file
nnoremap <leader>w :resize 65<CR>
