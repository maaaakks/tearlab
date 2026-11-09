# Provenance and third-party notices

TearLab derives from [Sepzie/Binding of AI](https://github.com/Sepzie/binding-of-ai),
originally authored by Sepehr Zohoori Rad. The existing Git history and author
credits are retained; source hashes and the source revision are in
`PUBLIC_SNAPSHOT.json`. Project-owned code, documentation and the bundled latest
checkpoint are dedicated under CC0-1.0; see `LICENSE`. This does not relicense
third-party components or alter the rights attached to historical versions.

The upstream repository does not declare a global license as of 2026-11-09.
The CC0 dedication applies to project-owned contributions only, and does not
assert ownership or grant a new license over inherited third-party material.

The broader local experiments tested the community
[nicklashansen/dreamer4](https://github.com/nicklashansen/dreamer4) implementation
under its own MIT license. That external checkout and its weights are not
included in this public tree. [RocketGPU](https://github.com/thecosmicskye/RocketGPU)
was consulted as a GPU simulation reference; this snapshot uses its own Isaac
physics kernels and does not bundle RocketGPU.

The Binding of Isaac and its characters belong to their respective owners.
TearLab is an unofficial research project. The README banner is AI-generated
fan artwork created with OpenAI image_gen, not a game screenshot or game asset.

The Lua JSON helper identifies itself as based on
[rxi/json.lua](https://github.com/rxi/json.lua), distributed under MIT. The
upstream notice is retained below. PyTorch, NumPy, W&B and LuaSocket are external
dependencies with their own licenses; their binaries are not bundled here.

`mod/preview_input.lua` is a new standalone CC0 adapter for this preview. The
legacy upstream `action_injector.lua` is not distributed in the current tree.

## rxi/json.lua - MIT

Copyright (c) 2020 rxi

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
