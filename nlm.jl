### A Pluto.jl notebook ###
# v0.12.2

using Markdown
using InteractiveUtils

# ╔═╡ 203e2102-0177-11eb-3707-4d76affe5f5a
begin
	using Images
	using TestImages
	using Statistics
	using StatsBase
	using BenchmarkTools
	using StatsPlots
end

# ╔═╡ 795f737a-097a-11eb-27f8-5b53a321031f
md"""
## Import image into notebook
"""

# ╔═╡ ae7dbf1e-03b0-11eb-3477-5795f954184b
begin
	img = testimage("cameraman")
	img = imresize(img, 256, 256)
end

# ╔═╡ 3170dec0-0789-11eb-0a01-5f3c3a9d901b
m, n = size(img)

# ╔═╡ c2409ddc-03b0-11eb-1ecb-7751f07fde0d
md"""
## Add Gaussian noise
"""

# ╔═╡ b61e4876-03b3-11eb-05fa-9bfba4f8e169
"""
Add guassian noise of standard deviation
σ (0-255) to `img`
"""
function add_gaussian_noise(img, σ)
	m, n = size(img)
	return img + Gray.(randn((m, n)) .* σ/255)
end

# ╔═╡ eb334944-078c-11eb-35bf-05c922292b81
noisy_img = add_gaussian_noise(img, 20)

# ╔═╡ 7fa88406-07db-11eb-3e33-9768637a3409
md"""
## Compute Standard deviation of noise
"""

# ╔═╡ 91ea7098-07db-11eb-3214-8d719c29aa94
function std_dev_of_noise(noisy_img, patch_size=3)
	noisy_mat = 255 .* Float64.(noisy_img)
	m, n = size(noisy_img)

	stds = []
	for i in 1+patch_size:m-patch_size
		for j in 1+patch_size:n-patch_size
			patch = noisy_mat[i-patch_size:i+patch_size, j-patch_size:j+patch_size]
			push!(stds, floor(Int32, std(patch)))
		end
	end

	return mode(stds)
end

# ╔═╡ 4b40f998-07dc-11eb-284c-abee0391c468
std_dev_of_noise(noisy_img)

# ╔═╡ 8d93bf7c-078b-11eb-0b8f-8b65ac6956a2
md"""
## Compute PSNR metric
"""

# ╔═╡ 9e3cf782-078b-11eb-213d-5d7ff86ef833
function PSNR(img, noisy_img)
	m, n = size(img)
	if (m, n) != size(noisy_img)
		throw(ArgumentError("Ground truth and noisy image don't have same dims"))
	end

	mse = (1/(m*n)) * sum((Float64.(img) - Float64.(noisy_img)).^2)

	return 20*log(10, 1/sqrt(mse))
end

# ╔═╡ f3f39112-078c-11eb-2c32-e1874399e05f
PSNR(img, noisy_img)

# ╔═╡ 0a1a9532-078e-11eb-1c30-bd39bfd5eb8b
md"""
## 1) NLM: Non Local Means

Original NLM implementation proposed by Buades et al.

[https://ieeexplore.ieee.org/document/1467423](https://ieeexplore.ieee.org/document/1467423)
"""

# ╔═╡ 5b346736-08ab-11eb-3df9-cb7d351e3b7b
md"""
#### Helper methods (hidden)
"""

# ╔═╡ 9053a966-07a2-11eb-2aaf-05fbc150b44c
# Helper functions
begin

	function gaussian_kernel(kernel_size)
		half_length = kernel_size ÷ 2
		kernel = zeros(kernel_size, kernel_size)
		sum = 0

		σ = 1

		for i in 1:kernel_size, j in 1:kernel_size
			x = i - half_length - 1
			y = j - half_length - 1

			# kernel[i, j] = (1/(2π*σ^2))*exp(-(x^2 + y^2)/(2σ^2))
			kernel[i, j] = exp(-(x^2 + y^2)/(2σ))
			sum += kernel[i, j]
		end

		return kernel ./ sum
	end

	function euclidean_distance_with_gaussian_kernel(patch1, patch2)
		(m, n) = size(patch1) # m and n should be of same size

		return sum(gaussian_kernel(m)*(patch1 - patch2).^2)
	end

	function euclidean_distance(patch1, patch2)
		N = prod(size(patch1))
		return (1/N)*sum(((patch1 - patch2).^2))
	end

	function get_patch(mat, i, j, patch_size)
		return mat[i-patch_size:i+patch_size, j-patch_size:j+patch_size]
	end

	function compare_patches(mat, i1, j1, i2, j2, patch_size, h, with_gauss=false)
		patch1 = get_patch(mat, i1, j1, patch_size)
		patch2 = get_patch(mat, i2, j2, patch_size)

		dist2 = 0

		if with_gauss
			dist2 = euclidean_distance_with_gaussian_kernel(patch1, patch2)
		else
			dist2 = euclidean_distance(patch1, patch2)
		end

		return exp(-dist2/h^2)
	end

	"""
	Extends the image border symmetrically
	"""
	function extend_img_border(img, ext_size)
		m, n = size(img)
		m = m+2*ext_size
		n = n+2*ext_size

		ext_img = Gray.(zeros(m, n))
		ext_img[1+ext_size:m-ext_size, 1+ext_size:n-ext_size] = img

		for i in 1:m
			mirror_pixel = ext_size
			for k in ext_size-1:-1:1
				ext_img[i, k] = ext_img[i, mirror_pixel]
				mirror_pixel += 1
			end
		end

		for j in 1:n
			mirror_pixel = ext_size
			for k in ext_size-1:-1:1
				ext_img[k, j] = ext_img[mirror_pixel, j]
				mirror_pixel += 1
			end
		end

		# Note that the 8 corder edges are also taken care of

		return ext_img
	end

	function clip_img_border(img, clip_size)
		m, n = size(img)
		return img[1+clip_size:m-clip_size, 1+clip_size:n-clip_size]
	end

	"""
	Check if pixel (i, j) is in border of the image or not

	The image border is of size border_size.

	NOTE: Will also return true if index is out of image
	"""
	function is_index_in_border(i, j, m, n, border_size)
		return i ≤ border_size || i ≥ m-border_size+1 ||
			   j ≤ border_size || j ≥ n-border_size+1
	end
end

# ╔═╡ 67adf39c-08ab-11eb-1951-91a0419354dc
md"""
#### Implementation

σ is taken to be 1, and `h` value is chose by trial and error
"""

# ╔═╡ 1656cdf2-078e-11eb-3c4e-d5e14f6e01e8
"""
search_window_size and similarity_window_size are the "half_lenghts"

Actual size of search window is 2*search_window_size+1

h is the smoothing factor
"""
function nlm(noisy_img, with_gauss_kernel=false, h=std_dev_of_noise(noisy_img),
			 search_window_size=10, similarity_window_size=2)
	# Extend noisy_img
	ext_noisy_img = extend_img_border(noisy_img, similarity_window_size)

	# Convert imamge to matrix
	noisy_mat = 255 .* Float64.(ext_noisy_img)
	m, n = size(noisy_mat)
	denoised_mat = zeros(m, n)

	# === Main Loop ===
	Threads.@threads for index in 0:(m*n-1)

		i = index÷n + 1
		j = index%n + 1

		# Ignore image border
		if is_index_in_border(i, j, m, n, similarity_window_size)
			denoised_mat[i, j] = noisy_mat[i, j]
			continue
		end

		# Weights matrix
		w = zeros((2*search_window_size+1, 2*search_window_size+1))

		# Search window margins
		above_margin = max(i - search_window_size, 1 + similarity_window_size)
		below_margin = min(i + search_window_size, m - similarity_window_size)
		left_margin  = max(j - search_window_size, 1 + similarity_window_size)
		right_margin = min(j + search_window_size, n - similarity_window_size)

		# Compute weights
		for si in above_margin:below_margin
			for sj in left_margin:right_margin
				wi = si-i+search_window_size+1
				wj = sj-j+search_window_size+1
				w[wi, wj] = compare_patches(noisy_mat, i, j, si, sj,
							similarity_window_size, h, with_gauss_kernel)
			end
		end

		# Normalize weights
		w ./= sum(w)

		# Update denoised image pixel
		for si in above_margin:below_margin
			for sj in left_margin:right_margin
				weight = w[si-i+search_window_size+1, sj-j+search_window_size+1]
				denoised_mat[i, j] += weight * noisy_mat[si, sj]
			end
		end
	end

	# Convert from matrix to image
	denoised_img = Gray.(denoised_mat ./255)

	# Clip off added symmetric border
	return clip_img_border(denoised_img, similarity_window_size)
end

# ╔═╡ 8c00cf1a-097a-11eb-0bbc-9d1cfe2297db
md"""
#### Denoised image
"""

# ╔═╡ 10301154-07a7-11eb-1bb1-f38aeaa17424
denoised_img = nlm(noisy_img, true, 2.5*std_dev_of_noise(noisy_img))

# ╔═╡ 9257a0be-097a-11eb-075a-a559aaca217a
md"""
#### Metrics
"""

# ╔═╡ 51eaa03e-07cd-11eb-0321-bf8067516643
PSNR(img, denoised_img)

# ╔═╡ d96570c4-07dd-11eb-1fcf-057985ae8b90
std_dev_of_noise(denoised_img)

# ╔═╡ 9e7fa5b2-097a-11eb-337d-bd263dbd1e5e
md"""
##### Original image, Nosiy image and Denoised image
"""

# ╔═╡ b98f8030-07d7-11eb-0945-e9502dfb88ae
mosaicview(img, noisy_img, denoised_img, nrow=1)

# ╔═╡ 8399f45e-0930-11eb-1dfd-39839851633f
md"""
## 2) NLM without Gaussian Kernel

The Guassian kernel while computing Euclidian distance is mostly dropped in several implementations of NLM (eg. MATLAB). A plausible reason for this is that the improvement in image quality is very litte compared to the additional parameters like σ that need to be tuned.

"""

# ╔═╡ de135470-0930-11eb-19f0-8fcf6f111c06
md"""
#### Implementation

Note that the implementation is similar to above with the main difference being in euclidean distance computation.
"""

# ╔═╡ 5059fb18-0979-11eb-20a5-33a62ccd7261
md"""
#### Denoised image
"""

# ╔═╡ 0e50b4b6-0931-11eb-3a73-7d3f70367efb
denoised_img1 = nlm(noisy_img, false, std_dev_of_noise(noisy_img))

# ╔═╡ 5b027914-0979-11eb-0630-0f9aa7c4679c
md"""
#### Metrics
"""

# ╔═╡ 49d65360-0931-11eb-224d-ffe1cb01a451
PSNR(img, denoised_img1)

# ╔═╡ 4d260038-0931-11eb-3da6-83616ce1a12c
std_dev_of_noise(denoised_img)

# ╔═╡ 0e5b96c6-097a-11eb-3e01-21121c1ce2ee
md"""
##### Original image, Nosiy image and Denoised image
"""

# ╔═╡ 53b3816e-0931-11eb-0c27-11e839a3fc9b
mosaicview(img, noisy_img, denoised_img1, nrow=1)

# ╔═╡ b55da7c8-08aa-11eb-20c3-ff7eb73b1aac
md"""
## 3) Fast NLM using Integral images

In this approach, the calculation of weights between pixels (`w(i, j)`) is improved by using integral images (aka summed area tables).

Note that this is a faster implementation of Method 2 above, hence the final denoised image obtained will be similar to that obtained from Method 2, as confirmed below.

Originally proposed: [https://ieeexplore.ieee.org/document/4541250/](https://ieeexplore.ieee.org/document/4541250/)

Implementation details (section 3.1): [https://www.ipol.im/pub/art/2014/120/](https://www.ipol.im/pub/art/2014/120/)
"""

# ╔═╡ a3f7a666-08ad-11eb-237f-37cb36a230a6
md"""
#### Helper methods (hidden)
"""

# ╔═╡ aa92aade-08ad-11eb-3a84-e5fc405f3300
begin
	function shifted_squared_norm(mat, z, t)
		diff = mat[z[1], z[2]]

		m, n = size(mat)
		if 1 ≤ z[1] + t[1] ≤ m && 1 ≤ z[2] + t[2] ≤ m
			diff -= mat[z[1]+t[1], z[2]+t[2]]
		end

		return diff^2
	end

	function compute_shifted_integral_matrix(mat, t)
		m, n = size(mat)
		St = zeros((m, n))

		St[1, 1] = shifted_squared_norm(mat, (1, 1), t)

		for x1 in 2:m
			St[x1, 1] = shifted_squared_norm(mat, (x1, 1), t) + St[x1-1, 1]
		end
		for x2 in 2:n
			St[1, x2] = shifted_squared_norm(mat, (1, x2), t) + St[1, x2-1]
		end

		for x1 in 2:m
			for x2 in 2:n
				St[x1, x2] = shifted_squared_norm(mat, (x1, x2), t) +
							 St[x1-1, x2] + St[x1, x2-1] - St[x1-1, x2-1]
			end
		end

		return St
	end

	function box_sum(St, x, ds)
		xi = x[1]
		xj = x[2]

		res = St[xi+ds, xj+ds]

		if xi-ds-1 != 0
			res -= St[xi-ds-1, xj+ds]
		end

		if xj-ds-1 != 0
			res -= St[xi+ds, xj-ds-1]
		end

		if xi-ds-1 != 0 && xj-ds-1 != 0
			res += St[xi-ds-1, xj-ds-1]
		end

		return res
	end
end

# ╔═╡ beaf8b50-08ac-11eb-0213-d1ef07894024
function fast_nlm(noisy_img, h=std_dev_of_noise(noisy_img),
				  search_window_size=10, similarity_window_size=2)

	# Extend noisy_img
	ext_noisy_img = extend_img_border(noisy_img, similarity_window_size)

	# Convert imamge to matrix
	noisy_mat = 255 .* Float64.(ext_noisy_img)
	m, n = size(noisy_mat)
	denoised_mat = zeros(m, n)

	# Sum of weights matrix
	sum_w = zeros((m, n))

	# === Main Loop ===
	offsets = -search_window_size:search_window_size
	for t1 in offsets, t2 in offsets

		# t shifted integral matrix
		St = compute_shifted_integral_matrix(noisy_mat, (t1, t2))

		Threads.@threads for index in 0:(m*n-1)
			# Assign indices for central pixel
			xi = index÷n + 1
			xj = index%n + 1

			# Coordinates to compute weights with
			yi = xi + t1
			yj = xj + t2

			# Ignore image border
			is_x_in_border = is_index_in_border(xi, xj, m, n, similarity_window_size)
			is_y_in_border = is_index_in_border(yi, yj, m, n, similarity_window_size)
			if is_x_in_border || is_y_in_border
				continue
			end

			# Compute euclidean distance between x and y
			dist2 = box_sum(St, (xi, xj), similarity_window_size)
			dist2 /= (2*similarity_window_size+1)^2

			# Compute weight
			weight = exp(-dist2/h^2)

			# Add weighted pixel to x (will be normalized later)
			denoised_mat[xi, xj] += weight * noisy_mat[yi, yj]

			# Update sum of weights
			sum_w[xi, xj] += weight
		end
	end

	# Normalize denoised_mat
	for index in 0:(m*n-1)
		i = index÷n + 1
		j = index%n + 1
		if sum_w[i, j] != 0
			denoised_mat[i, j] /= sum_w[i, j]
		end
	end

	# Convert from matrix to image
	denoised_img = Gray.(denoised_mat ./255)

	# Clip off added symmetric border
	return clip_img_border(denoised_img, similarity_window_size)

end

# ╔═╡ ca43d29c-0978-11eb-263f-17a1c3640db5
md"""
#### Denoised image
"""

# ╔═╡ 29489ec6-08b6-11eb-092b-f708625b8b70
denoised_img2 = fast_nlm(noisy_img)

# ╔═╡ e856200c-0978-11eb-1db8-55811d666d37
md"""
#### Metrics
"""

# ╔═╡ 5c480932-08c6-11eb-1e3f-b580cbbadbee
PSNR(img, denoised_img2)

# ╔═╡ 63330d1e-08c6-11eb-07e5-7d8c15de30e7
std_dev_of_noise(denoised_img2)

# ╔═╡ f3076518-0978-11eb-30a1-8b596a069dae
md"""
##### Original image, Nosiy image and Denoised image
"""

# ╔═╡ 640c46d8-08c6-11eb-2424-e3474293e6c2
mosaicview(img, noisy_img, denoised_img, nrow=1)

# ╔═╡ 68e87226-08c6-11eb-3b1f-830ae7867aff
# Both methods should give the same result
denoised_img2 - denoised_img1

# ╔═╡ 5df67eda-093e-11eb-0d5e-79efe0b75ee4
md"""
## Results
"""

# ╔═╡ 2c384944-0963-11eb-1ed7-494102639ce2
md"""
Noisy images with sigma 5, 10, 15, 20 and 25
"""

# ╔═╡ 68ebb94a-093e-11eb-3b04-65adeeda65a5
begin
	noisy_imgs = [add_gaussian_noise(img, sigma) for sigma in 5:5:25]
	mosaicview(noisy_imgs, nrow=1)
	noisy_PSNRs = [PSNR(img, noisy_img) for noisy_img in noisy_imgs]
end

# ╔═╡ 981a04d0-0964-11eb-3b84-6f8a706c9c74
begin
	denoised_imgs = [Gray.(zeros(m, n)) for i in 1:5, j in 1:3]
	PSNRs = zeros(5, 3)
	noise_std_devs = zeros(5, 3)
end

# ╔═╡ b0928538-0963-11eb-3a49-4fef415e3c17
begin

	for i in 1:5

		# Method 1
		denoised_imgs[i, 1] = nlm(noisy_imgs[i], true,
								 2.5*std_dev_of_noise(noisy_imgs[i]))
		println("Method 1 done.")

		# Method 2
		denoised_imgs[i, 2] = nlm(noisy_imgs[i], false,
								 std_dev_of_noise(noisy_imgs[i]))
		println("Method 2 done.")

		# Method 3
		denoised_imgs[i, 3] = fast_nlm(noisy_imgs[i])
		println("Method 3 done.")

		# Compute PSNR and noise std deviations of denoised image
		for j in 1:3
			PSNRs[i, j] = PSNR(img, denoised_imgs[i, j])
			noise_std_devs[i, j] = std_dev_of_noise(denoised_imgs[i, j])
		end

		println("Noisy img with sigma = $(5*i) done.")
		println()
	end

end

# ╔═╡ 90ff81f8-0986-11eb-1e39-b1b1658bebef
begin
	noisy_mosaic = mosaicview(noisy_imgs, nrow=1)
	save("noisy_mosaic.png", colorview(Gray, noisy_mosaic))
end

# ╔═╡ cbe561c8-096a-11eb-0fa3-4987532d2da7
for i in 1:5
	mosaic = mosaicview(denoised_imgs[i, :], nrow=1)
	save("mosaic-$(5*i).png", colorview(Gray, mosaic))
end

# ╔═╡ d439af7a-097a-11eb-2db1-6f903b5a6c57
begin
	xticks = (1:5, 5:5:25)
	data = [noisy_PSNRs PSNRs]
	labels = repeat(["Noisy", "NLM", "NLM without Gaussian", "Fast NLM"], inner=5)
	ylabel = "PSNR"
	xlabel = "Noise Standard Deviation"
	title = ""
	groupedbar(data, ylabel=ylabel, xlabel=xlabel, title=title, group=labels, lw=0, framestyle= :box, legend=(0.73, 0.93), ylim=(0,45), xticks=xticks)
	savefig("plot")
end

# ╔═╡ 413f8862-0980-11eb-3c60-ab97aeeeaa22


# ╔═╡ eac6288a-07d4-11eb-347c-dfc4987a6629
md"""
#### Extras
"""

# ╔═╡ f6082068-07d4-11eb-1dac-7f72f40795e8
function extend_mat(M::AbstractMatrix, i, j)
	h, w = size(M)
	if 1 ≤ i ≤ h && 1 ≤ j ≤ w
		return M[i, j]
	elseif 1 ≤ i ≤ h
		if j < 1
			return M[i, 1]
		else
			return M[i, w]
		end
	elseif 1 ≤ j ≤ w
		if i < 1
			return M[1, j]
		else
			return M[h, j]
		end
	else
		if i < 1 && j < 1
			return M[1, 1]
		elseif i < 1 && j > w
			return M[1, w]
		elseif i > h && j > w
			return M[h, w]
		else
			return M[h, 1]
		end
	end
end

# ╔═╡ Cell order:
# ╠═203e2102-0177-11eb-3707-4d76affe5f5a
# ╟─795f737a-097a-11eb-27f8-5b53a321031f
# ╠═ae7dbf1e-03b0-11eb-3477-5795f954184b
# ╠═3170dec0-0789-11eb-0a01-5f3c3a9d901b
# ╟─c2409ddc-03b0-11eb-1ecb-7751f07fde0d
# ╠═b61e4876-03b3-11eb-05fa-9bfba4f8e169
# ╠═eb334944-078c-11eb-35bf-05c922292b81
# ╟─7fa88406-07db-11eb-3e33-9768637a3409
# ╠═91ea7098-07db-11eb-3214-8d719c29aa94
# ╠═4b40f998-07dc-11eb-284c-abee0391c468
# ╟─8d93bf7c-078b-11eb-0b8f-8b65ac6956a2
# ╠═9e3cf782-078b-11eb-213d-5d7ff86ef833
# ╠═f3f39112-078c-11eb-2c32-e1874399e05f
# ╟─0a1a9532-078e-11eb-1c30-bd39bfd5eb8b
# ╟─5b346736-08ab-11eb-3df9-cb7d351e3b7b
# ╠═9053a966-07a2-11eb-2aaf-05fbc150b44c
# ╟─67adf39c-08ab-11eb-1951-91a0419354dc
# ╠═1656cdf2-078e-11eb-3c4e-d5e14f6e01e8
# ╟─8c00cf1a-097a-11eb-0bbc-9d1cfe2297db
# ╠═10301154-07a7-11eb-1bb1-f38aeaa17424
# ╟─9257a0be-097a-11eb-075a-a559aaca217a
# ╠═51eaa03e-07cd-11eb-0321-bf8067516643
# ╠═d96570c4-07dd-11eb-1fcf-057985ae8b90
# ╟─9e7fa5b2-097a-11eb-337d-bd263dbd1e5e
# ╠═b98f8030-07d7-11eb-0945-e9502dfb88ae
# ╟─8399f45e-0930-11eb-1dfd-39839851633f
# ╟─de135470-0930-11eb-19f0-8fcf6f111c06
# ╠═5059fb18-0979-11eb-20a5-33a62ccd7261
# ╠═0e50b4b6-0931-11eb-3a73-7d3f70367efb
# ╟─5b027914-0979-11eb-0630-0f9aa7c4679c
# ╠═49d65360-0931-11eb-224d-ffe1cb01a451
# ╠═4d260038-0931-11eb-3da6-83616ce1a12c
# ╟─0e5b96c6-097a-11eb-3e01-21121c1ce2ee
# ╠═53b3816e-0931-11eb-0c27-11e839a3fc9b
# ╟─b55da7c8-08aa-11eb-20c3-ff7eb73b1aac
# ╟─a3f7a666-08ad-11eb-237f-37cb36a230a6
# ╟─aa92aade-08ad-11eb-3a84-e5fc405f3300
# ╠═beaf8b50-08ac-11eb-0213-d1ef07894024
# ╟─ca43d29c-0978-11eb-263f-17a1c3640db5
# ╠═29489ec6-08b6-11eb-092b-f708625b8b70
# ╟─e856200c-0978-11eb-1db8-55811d666d37
# ╠═5c480932-08c6-11eb-1e3f-b580cbbadbee
# ╠═63330d1e-08c6-11eb-07e5-7d8c15de30e7
# ╟─f3076518-0978-11eb-30a1-8b596a069dae
# ╠═640c46d8-08c6-11eb-2424-e3474293e6c2
# ╠═68e87226-08c6-11eb-3b1f-830ae7867aff
# ╟─5df67eda-093e-11eb-0d5e-79efe0b75ee4
# ╟─2c384944-0963-11eb-1ed7-494102639ce2
# ╠═68ebb94a-093e-11eb-3b04-65adeeda65a5
# ╠═981a04d0-0964-11eb-3b84-6f8a706c9c74
# ╠═b0928538-0963-11eb-3a49-4fef415e3c17
# ╠═90ff81f8-0986-11eb-1e39-b1b1658bebef
# ╠═cbe561c8-096a-11eb-0fa3-4987532d2da7
# ╠═d439af7a-097a-11eb-2db1-6f903b5a6c57
# ╟─413f8862-0980-11eb-3c60-ab97aeeeaa22
# ╟─eac6288a-07d4-11eb-347c-dfc4987a6629
# ╟─f6082068-07d4-11eb-1dac-7f72f40795e8
